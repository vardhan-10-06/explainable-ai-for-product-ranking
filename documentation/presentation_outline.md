# Presentation Outline
*(Detailed slide-by-slide outline. Import into PowerPoint/Google Slides/Beamer --
ships as Markdown here rather than a compiled .pptx.)*

**Slide 1 -- Title**
Explainable AI for Product Ranking Using Hybrid Machine Learning, Knowledge Graphs,
and Large Language Models. [Author names, affiliation, date]

**Slide 2 -- Motivation**
- Ranking systems are high-impact but opaque
- Trust gap: users can't tell *why* an item ranked highly; operators can't audit it
- Goal: rank well AND explain every decision, with verified explanations

**Slide 3 -- Problem statement**
Learning-to-rank formulation: f(user, product) -> relevance score, with the added
requirement that every score be explainable and the explanation be verifiable.

**Slide 4 -- System architecture (diagram)**
Reproduce the component diagram from `documentation/architecture.md`.

**Slide 5 -- Hybrid ranking ensemble**
LightGBM + XGBoost + Neural Ranker + DeepFM -> weighted ensemble. One line per model
on why it's included (complementary inductive biases).

**Slide 6 -- Knowledge graph**
Entity/relationship diagram (Users, Products, Categories, Brands, Reviews;
PURCHASED/VIEWED/BELONGS_TO/SIMILAR_TO/LIKED/REVIEWED/BOUGHT_WITH). Used both as
ranking features (degree, PageRank, embeddings) and as explanation evidence (paths).

**Slide 7 -- Explainability layer**
SHAP local/global explanations; RankSHAP-style pairwise "why A over B" comparison.

**Slide 8 -- LLM explanation generation + hallucination detection**
Pipeline diagram: evidence -> prompt -> LLM -> verification -> final explanation (with
deterministic fallback on verification failure).

**Slide 9 -- The Explainability Confidence Score (ECS)**
Formula slide: ECS = w_F F + w_G G + w_L L + w_V V + w_C C. One-line definition of
each component.

**Slide 10 -- Evaluation methodology**
Ranking metrics, explainability metrics, NLP metrics, efficiency metrics -- one row
each in a summary table.

**Slide 11 -- Experimental setup**
Dataset, split strategy, baselines, ablation design, significance testing.

**Slide 12 -- Results (placeholder)**
Bar chart: hybrid ensemble vs. popularity baseline vs. ablated variants on NDCG@10
(populate from `results/ranking_comparison.png` / `results/ablation_comparison.png`).

**Slide 13 -- Demo**
Screenshot(s) of the Streamlit dashboard: search -> rankings -> explanation with SHAP
chart and KG path visualization -> ECS score.

**Slide 14 -- Limitations and future work**
Synthetic-data caveat, lexical-overlap detector trade-off, fixed ECS weights,
real-dataset validation as next step.

**Slide 15 -- Conclusion**
Recap contributions: hybrid ensemble, KG-grounded explanations, verified LLM
explanations, novel ECS metric, full open-source reference implementation.

**Slide 16 -- Q&A / References**
