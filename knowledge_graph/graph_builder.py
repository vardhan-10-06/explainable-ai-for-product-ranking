"""
knowledge_graph/graph_builder.py
==================================
Builds the product-ranking knowledge graph.

Nodes: User, Product, Category, Brand, Review, Attribute
Relationships: PURCHASED, VIEWED, BELONGS_TO, SIMILAR_TO, LIKED, REVIEWED,
               BOUGHT_WITH

Two backends are supported:
  * "networkx" (default) - in-memory graph, zero external services, good
     for research/reproducibility and for running this project without
     standing up infrastructure.
  * "neo4j" - production-grade graph database. When selected, the same
     logical graph is written via Cypher (see cypher_queries.cql for the
     canonical query set) so the exact same relationships/traversals are
     available in a real deployment.

Trade-off: NetworkX cannot do server-side Cypher traversal at scale, but
needs no setup; Neo4j scales and supports rich graph algorithms (GDS
library) but requires a running server. We default to NetworkX so the
whole pipeline is runnable out of the box, and provide a thin Neo4jGraph
adapter with the same public interface so switching backends is a one-line
config change (`knowledge_graph.backend: neo4j`).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from common.config import get_config, get_logger

logger = get_logger(__name__)


class KnowledgeGraphBuilder:
    """Builds and queries the product-ranking knowledge graph (NetworkX backend)."""

    def __init__(self):
        self.cfg = get_config()["knowledge_graph"]
        self.graph = nx.MultiDiGraph()

    def build(
        self,
        users: pd.DataFrame,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
        reviews: pd.DataFrame,
    ) -> nx.MultiDiGraph:
        """
        Construct the full graph from processed tables.

        Args:
            users, products, interactions, reviews: Processed common-schema
                tables (see preprocessing.preprocess.ProcessedData).

        Returns:
            The populated NetworkX MultiDiGraph.
        """
        self._add_user_nodes(users)
        self._add_product_nodes(products)
        self._add_category_and_brand_nodes(products)
        self._add_review_nodes(reviews)
        self._add_interaction_edges(interactions)
        self._add_bought_with_edges(interactions)
        self._add_similar_to_edges(products)

        logger.info(
            "Knowledge graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(), self.graph.number_of_edges(),
        )
        return self.graph

    # ------------------------------------------------------------------ #
    def _add_user_nodes(self, users: pd.DataFrame) -> None:
        for row in users.itertuples(index=False):
            self.graph.add_node(f"User:{row.user_id}", type="User", label=row.user_id)

    def _add_product_nodes(self, products: pd.DataFrame) -> None:
        for row in products.itertuples(index=False):
            self.graph.add_node(
                f"Product:{row.product_id}", type="Product", label=row.product_id,
                price=getattr(row, "price", None), avg_rating=getattr(row, "avg_rating", None),
            )

    def _add_category_and_brand_nodes(self, products: pd.DataFrame) -> None:
        for category in products["category"].unique():
            self.graph.add_node(f"Category:{category}", type="Category", label=category)
        for brand in products["brand"].unique():
            self.graph.add_node(f"Brand:{brand}", type="Brand", label=brand)
        for row in products.itertuples(index=False):
            self.graph.add_edge(f"Product:{row.product_id}", f"Category:{row.category}", relation="BELONGS_TO")
            self.graph.add_edge(f"Product:{row.product_id}", f"Brand:{row.brand}", relation="BELONGS_TO")

    def _add_review_nodes(self, reviews: pd.DataFrame) -> None:
        if reviews.empty:
            return
        for row in reviews.itertuples(index=False):
            review_node = f"Review:{row.review_id}"
            self.graph.add_node(review_node, type="Review", label=row.review_id, rating=row.rating, text=row.text)
            self.graph.add_edge(f"User:{row.user_id}", review_node, relation="REVIEWED")
            self.graph.add_edge(review_node, f"Product:{row.product_id}", relation="REVIEWED")
            if row.rating is not None and row.rating >= 4:
                self.graph.add_edge(f"User:{row.user_id}", f"Product:{row.product_id}", relation="LIKED")

    def _add_interaction_edges(self, interactions: pd.DataFrame) -> None:
        relation_map = {"purchase": "PURCHASED", "view": "VIEWED", "wishlist": "LIKED", "cart": "VIEWED"}
        for row in interactions.itertuples(index=False):
            relation = relation_map.get(row.event_type, "VIEWED")
            self.graph.add_edge(
                f"User:{row.user_id}", f"Product:{row.product_id}", relation=relation, days_ago=row.days_ago,
            )

    def _add_bought_with_edges(self, interactions: pd.DataFrame) -> None:
        """Co-purchase edges: products purchased by the same user within the same session-ish window."""
        purchases = interactions[interactions["event_type"] == "purchase"]
        for _, group in purchases.groupby("user_id"):
            products_bought = group["product_id"].tolist()
            for i in range(len(products_bought)):
                for j in range(i + 1, len(products_bought)):
                    a, b = products_bought[i], products_bought[j]
                    if a != b:
                        self.graph.add_edge(f"Product:{a}", f"Product:{b}", relation="BOUGHT_WITH")
                        self.graph.add_edge(f"Product:{b}", f"Product:{a}", relation="BOUGHT_WITH")

    def _add_similar_to_edges(self, products: pd.DataFrame) -> None:
        """Products in the same category+brand are considered SIMILAR_TO one another."""
        for (_, _), group in products.groupby(["category", "brand"]):
            ids = group["product_id"].tolist()
            for i in range(len(ids)):
                for j in range(i + 1, min(i + 6, len(ids))):  # cap fan-out for tractability
                    self.graph.add_edge(f"Product:{ids[i]}", f"Product:{ids[j]}", relation="SIMILAR_TO")
                    self.graph.add_edge(f"Product:{ids[j]}", f"Product:{ids[i]}", relation="SIMILAR_TO")

    # ------------------------------------------------------------------ #
    def compute_graph_features(self) -> pd.DataFrame:
        """
        Compute degree centrality and PageRank per Product node as
        graph-derived ranking features.

        Returns:
            DataFrame indexed by product_id with columns
            graph_degree, graph_pagerank.
        """
        product_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Product"]
        subgraph = self.graph.subgraph(product_nodes + [
            n for n in self.graph.nodes if n not in product_nodes
        ])
        degree = dict(self.graph.degree(product_nodes))
        try:
            pagerank = nx.pagerank(nx.DiGraph(self.graph), max_iter=100)
        except nx.PowerIterationFailedConvergence:  # pragma: no cover
            pagerank = {n: 0.0 for n in self.graph.nodes}

        rows = []
        for node in product_nodes:
            product_id = node.split(":", 1)[1]
            rows.append({
                "product_id": product_id,
                "graph_degree": degree.get(node, 0),
                "graph_pagerank": pagerank.get(node, 0.0),
            })
        return pd.DataFrame(rows).set_index("product_id")

    def compute_graph_embeddings(self, dim: Optional[int] = None) -> pd.DataFrame:
        """
        Compute lightweight graph embeddings for Product nodes via biased
        random walks + a co-occurrence factorization (a dependency-free
        node2vec-style approximation, avoiding a heavy gensim/word2vec
        dependency for this reference implementation).

        Args:
            dim: Embedding dimensionality; defaults to config value.

        Returns:
            DataFrame indexed by product_id with columns kg_emb_0..kg_emb_{dim-1}.
        """
        dim = dim or self.cfg["embedding_dim"]
        walk_length = self.cfg["node2vec_walk_length"]
        num_walks = self.cfg["node2vec_num_walks"]

        product_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == "Product"]
        if not product_nodes:
            return pd.DataFrame(columns=[f"kg_emb_{i}" for i in range(dim)])

        walks = self._generate_random_walks(product_nodes, walk_length, num_walks)
        cooccurrence = self._build_cooccurrence(walks, product_nodes)
        embeddings = self._factorize(cooccurrence, dim)

        cols = [f"kg_emb_{i}" for i in range(dim)]
        emb_df = pd.DataFrame(embeddings, columns=cols)
        emb_df.insert(0, "product_id", [n.split(":", 1)[1] for n in product_nodes])
        return emb_df.set_index("product_id")

    def _generate_random_walks(self, product_nodes: List[str], walk_length: int, num_walks: int) -> List[List[str]]:
        rng = np.random.default_rng(get_config()["project"]["seed"])
        walks = []
        undirected = self.graph.to_undirected()
        for node in product_nodes:
            for _ in range(num_walks):
                walk = [node]
                current = node
                for _ in range(walk_length - 1):
                    neighbors = list(undirected.neighbors(current))
                    if not neighbors:
                        break
                    current = neighbors[rng.integers(0, len(neighbors))]
                    walk.append(current)
                walks.append(walk)
        return walks

    def _build_cooccurrence(self, walks: List[List[str]], product_nodes: List[str]) -> sparse.csr_matrix:
        index = {node: i for i, node in enumerate(product_nodes)}
        n = len(product_nodes)
        rows, columns = [], []
        window = 3
        for walk in walks:
            product_positions = [(i, node) for i, node in enumerate(walk) if node in index]
            for i, (pos_i, node_i) in enumerate(product_positions):
                for pos_j, node_j in product_positions[i + 1:i + 1 + window]:
                    a, b = index[node_i], index[node_j]
                    rows.extend((a, b))
                    columns.extend((b, a))
        values = np.ones(len(rows), dtype=np.float32)
        return sparse.coo_matrix((values, (rows, columns)), shape=(n, n)).tocsr()

    def _factorize(self, matrix: np.ndarray, dim: int) -> np.ndarray:
        """Truncated SVD as a dependency-light stand-in for node embedding factorization."""
        if matrix.shape[0] == 0:
            return np.zeros((0, dim))
        if sparse.issparse(matrix):
            if matrix.shape[0] == 1:
                return np.zeros((1, dim))
            component_count = min(dim, matrix.shape[0] - 1)
            factorizer = TruncatedSVD(
                n_components=component_count,
                random_state=get_config()["project"]["seed"],
            )
            projected = factorizer.fit_transform(matrix)
            embedding = projected / np.sqrt(
                np.maximum(factorizer.singular_values_, np.finfo(np.float32).eps)
            )
            if embedding.shape[1] < dim:
                embedding = np.pad(embedding, ((0, 0), (0, dim - embedding.shape[1])))
            return embedding
        dim = min(dim, matrix.shape[0])
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        embedding = u[:, :dim] * np.sqrt(s[:dim] + 1e-6)
        if embedding.shape[1] < dim:
            embedding = np.pad(embedding, ((0, 0), (0, dim - embedding.shape[1])))
        return embedding

    # ------------------------------------------------------------------ #
    def explain_path(self, user_id: str, product_id: str, max_hops: int = 3) -> List[Dict]:
        """
        Find a short explanatory path between a user and a product in the
        graph (e.g. User -LIKED-> Product A -SIMILAR_TO-> Product B),
        used by the explainability/LLM modules to ground explanations in
        real graph evidence.

        Args:
            user_id: Source user id (without the "User:" prefix).
            product_id: Target product id (without the "Product:" prefix).
            max_hops: Maximum path length to search.

        Returns:
            List of edge dicts describing the path, or an empty list if no
            path within max_hops was found.
        """
        source, target = f"User:{user_id}", f"Product:{product_id}"
        if source not in self.graph or target not in self.graph:
            return []
        try:
            path = nx.shortest_path(self.graph.to_undirected(), source, target)
        except nx.NetworkXNoPath:
            return []
        if len(path) - 1 > max_hops:
            return []

        edges = []
        for a, b in zip(path[:-1], path[1:]):
            relation = "RELATED_TO"
            if self.graph.has_edge(a, b):
                edge_data = list(self.graph.get_edge_data(a, b).values())[0]
                relation = edge_data.get("relation", relation)
            edges.append({"from": a, "to": b, "relation": relation})
        return edges
