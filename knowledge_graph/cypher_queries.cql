// =============================================================================
// cypher_queries.cql
// Canonical Cypher queries for the Neo4j-backed knowledge graph.
// Use these when knowledge_graph.backend = "neo4j" in configs/config.yaml.
// Run with: cypher-shell -u neo4j -p <password> -f cypher_queries.cql
// or execute sections individually from the Neo4j Browser.
// =============================================================================

// ---------------------------------------------------------------------------
// 1. Constraints & indexes (run once)
// ---------------------------------------------------------------------------
CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;
CREATE CONSTRAINT product_id_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE;
CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT brand_name_unique IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE;
CREATE CONSTRAINT review_id_unique IF NOT EXISTS FOR (r:Review) REQUIRE r.review_id IS UNIQUE;

// ---------------------------------------------------------------------------
// 2. Node ingestion (expects CSVs staged under Neo4j's import/ directory)
// ---------------------------------------------------------------------------
LOAD CSV WITH HEADERS FROM 'file:///users.csv' AS row
MERGE (u:User {user_id: row.user_id})
SET u.signup_days_ago = toInteger(row.signup_days_ago),
    u.avg_session_len = toFloat(row.avg_session_len);

LOAD CSV WITH HEADERS FROM 'file:///products.csv' AS row
MERGE (p:Product {product_id: row.product_id})
SET p.price = toFloat(row.price),
    p.avg_rating = toFloat(row.avg_rating),
    p.num_reviews = toInteger(row.num_reviews)
MERGE (c:Category {name: row.category})
MERGE (b:Brand {name: row.brand})
MERGE (p)-[:BELONGS_TO]->(c)
MERGE (p)-[:BELONGS_TO]->(b);

LOAD CSV WITH HEADERS FROM 'file:///reviews.csv' AS row
MATCH (u:User {user_id: row.user_id})
MATCH (p:Product {product_id: row.product_id})
MERGE (r:Review {review_id: row.review_id})
SET r.rating = toFloat(row.rating), r.text = row.text
MERGE (u)-[:REVIEWED]->(r)
MERGE (r)-[:REVIEWED]->(p)
WITH u, p, row
FOREACH (_ IN CASE WHEN toFloat(row.rating) >= 4 THEN [1] ELSE [] END |
  MERGE (u)-[:LIKED]->(p)
);

// ---------------------------------------------------------------------------
// 3. Interaction edges (PURCHASED / VIEWED)
// ---------------------------------------------------------------------------
LOAD CSV WITH HEADERS FROM 'file:///interactions.csv' AS row
MATCH (u:User {user_id: row.user_id})
MATCH (p:Product {product_id: row.product_id})
FOREACH (_ IN CASE WHEN row.event_type = 'purchase' THEN [1] ELSE [] END |
  MERGE (u)-[rel:PURCHASED]->(p) SET rel.days_ago = toInteger(row.days_ago)
)
FOREACH (_ IN CASE WHEN row.event_type = 'view' THEN [1] ELSE [] END |
  MERGE (u)-[rel:VIEWED]->(p) SET rel.days_ago = toInteger(row.days_ago)
);

// ---------------------------------------------------------------------------
// 4. Derived relationships: BOUGHT_WITH (co-purchase) and SIMILAR_TO
// ---------------------------------------------------------------------------
MATCH (u:User)-[:PURCHASED]->(p1:Product), (u)-[:PURCHASED]->(p2:Product)
WHERE p1.product_id < p2.product_id
MERGE (p1)-[:BOUGHT_WITH]->(p2)
MERGE (p2)-[:BOUGHT_WITH]->(p1);

MATCH (p1:Product)-[:BELONGS_TO]->(c:Category)<-[:BELONGS_TO]-(p2:Product)
WHERE p1.product_id < p2.product_id
MERGE (p1)-[:SIMILAR_TO]->(p2)
MERGE (p2)-[:SIMILAR_TO]->(p1);

// ---------------------------------------------------------------------------
// 5. Example traversal queries used by the explainability / retrieval layer
// ---------------------------------------------------------------------------

// (a) Explain why a product might be recommended to a user (path evidence)
// :param user_id => "U000123"
// :param product_id => "P000045"
MATCH path = shortestPath(
  (u:User {user_id: $user_id})-[*..3]-(p:Product {product_id: $product_id})
)
RETURN path;

// (b) Products frequently bought together with a given product
MATCH (p:Product {product_id: $product_id})-[:BOUGHT_WITH]->(other:Product)
RETURN other.product_id, count(*) AS co_purchase_strength
ORDER BY co_purchase_strength DESC
LIMIT 10;

// (c) Category-based candidate generation for retrieval
MATCH (u:User {user_id: $user_id})-[:PURCHASED|LIKED]->(:Product)-[:BELONGS_TO]->(c:Category)
      <-[:BELONGS_TO]-(candidate:Product)
WHERE NOT (u)-[:PURCHASED]->(candidate)
RETURN DISTINCT candidate.product_id, candidate.avg_rating
ORDER BY candidate.avg_rating DESC
LIMIT 20;

// (d) Graph degree / popularity, usable as a ranking feature
MATCH (p:Product)
OPTIONAL MATCH (p)<-[r:PURCHASED|VIEWED]-()
RETURN p.product_id, count(r) AS interaction_degree
ORDER BY interaction_degree DESC;

// (e) Fact-check helper for the hallucination detector: verify a claimed
//     attribute (brand/category) about a product before it is surfaced in
//     an LLM-generated explanation.
MATCH (p:Product {product_id: $product_id})-[:BELONGS_TO]->(entity)
RETURN labels(entity) AS entity_type, entity.name AS entity_name;
