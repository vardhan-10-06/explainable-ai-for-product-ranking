# User Manual (Dashboard)

## 1. Log in
Enter your username/password in the sidebar and click "Log in". If you don't have an
account yet, register via the API's `/auth/register` endpoint (see API documentation)
or `POST` it with `curl`/Postman before logging into the dashboard.

## 2. Search tab
Type a free-text query (matches product ID, category, or brand) and click "Search" to
browse the catalog.

## 3. Rankings tab
Enter a User ID and click "Get Rankings" to see that user's personalized top-K ranked
products, with a bar chart of ranking scores. The ranked product list is remembered
so you can immediately explain any of them in the next tab.

## 4. Explanations tab
Pick a product from your last ranking result (or type a Product ID directly), and
optionally pick a second product to compare against. Click "Generate Explanation" to
see:
- A plain-language explanation of the ranking decision
- **Confidence Score**: how well the explanation passed fact-verification
- **ECS Score**: the overall Explainability Confidence Score (0-1)
- A SHAP bar chart of the top contributing features
- A visual knowledge-graph path connecting the user to the product

## 5. Evaluation tab
View the latest evaluation report (ranking metrics for the hybrid ensemble vs. a
popularity baseline) as both a JSON dump and a comparison bar chart. Run
`python -m evaluation.evaluate` first if this tab shows no data.
