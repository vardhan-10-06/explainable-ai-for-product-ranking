"""
knowledge_graph/neo4j_adapter.py
==================================
Thin adapter that runs the canonical Cypher queries (cypher_queries.cql)
against a live Neo4j instance, exposing the same `explain_path` /
`get_co_purchased` interface used elsewhere in the codebase so callers
don't need to care which backend (NetworkX vs Neo4j) is active.

This module is only imported when `knowledge_graph.backend: neo4j` in the
config; the `neo4j` Python driver is an optional dependency for exactly
this reason (kept out of the default NetworkX code path).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from common.config import get_config, get_logger

logger = get_logger(__name__)

CYPHER_FILE = Path(__file__).parent / "cypher_queries.cql"


class Neo4jGraphAdapter:
    """Production Neo4j-backed equivalent of KnowledgeGraphBuilder's query surface."""

    def __init__(self):
        try:
            from neo4j import GraphDatabase  # local import: optional dependency
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'neo4j' package is required for the Neo4j backend. "
                "Install with `pip install neo4j`."
            ) from exc

        cfg = get_config()["knowledge_graph"]["neo4j"]
        self._driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
        logger.info("Connected to Neo4j at %s", cfg["uri"])

    def close(self) -> None:
        """Close the underlying driver connection."""
        self._driver.close()

    def run_setup_script(self) -> None:
        """
        Execute the constraint/index/ingestion statements from
        cypher_queries.cql. Expects the CSVs referenced inside that file
        to already be staged in Neo4j's `import/` directory.
        """
        statements = [s.strip() for s in CYPHER_FILE.read_text().split(";") if s.strip() and not s.strip().startswith("//")]
        with self._driver.session() as session:
            for stmt in statements:
                if stmt.upper().startswith(("LOAD CSV", "CREATE CONSTRAINT")):
                    session.run(stmt)
        logger.info("Neo4j setup script executed (%d statements).", len(statements))

    def explain_path(self, user_id: str, product_id: str, max_hops: int = 3) -> List[Dict]:
        """Mirror of KnowledgeGraphBuilder.explain_path but backed by a live Cypher query."""
        query = f"""
        MATCH path = shortestPath((u:User {{user_id: $user_id}})-[*..{max_hops}]-(p:Product {{product_id: $product_id}}))
        RETURN [r IN relationships(path) | type(r)] AS relations,
               [n IN nodes(path) | coalesce(n.user_id, n.product_id, n.name)] AS node_labels
        """
        with self._driver.session() as session:
            result = session.run(query, user_id=user_id, product_id=product_id).single()
        if not result:
            return []
        relations, labels = result["relations"], result["node_labels"]
        return [
            {"from": labels[i], "to": labels[i + 1], "relation": relations[i]}
            for i in range(len(relations))
        ]

    def get_co_purchased(self, product_id: str, limit: int = 10) -> List[Dict]:
        """Fetch top co-purchased products for a given product."""
        query = """
        MATCH (p:Product {product_id: $product_id})-[:BOUGHT_WITH]->(other:Product)
        RETURN other.product_id AS product_id, count(*) AS strength
        ORDER BY strength DESC LIMIT $limit
        """
        with self._driver.session() as session:
            return [dict(r) for r in session.run(query, product_id=product_id, limit=limit)]
