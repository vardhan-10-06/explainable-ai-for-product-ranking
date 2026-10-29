"""Professional Streamlit workspace for explainable product ranking."""
from __future__ import annotations

import os
from typing import Any, Optional

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Rankwise | Explainable Product Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#172033; --muted:#667085; --blue:#356AE6; --soft:#F4F7FC; --line:#E6EAF2; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
    h1,h2,h3 { font-family:'Space Grotesk', sans-serif; letter-spacing:-.03em; }
    .stApp { background:linear-gradient(180deg,#F8FAFF 0%,#FFFFFF 34%); }
    [data-testid="stSidebar"] { background:#111A2E; border-right:0; }
    [data-testid="stSidebar"] * { color:#E9EEFA !important; }
    [data-testid="stSidebar"] .stTextInput input { background:#1D2942; border-color:#344361; }
    .brand { padding:1.2rem .5rem 1.8rem; }
    .brand-mark { display:inline-flex; width:38px; height:38px; align-items:center; justify-content:center;
      border-radius:12px; background:#5E8BFF; color:white; font-size:22px; font-weight:700; margin-right:10px; }
    .brand-name { font:700 21px 'Space Grotesk'; color:#fff; vertical-align:middle; }
    .brand-sub { color:#91A1C1; font-size:12px; margin:9px 0 0 48px; }
    .hero { padding:1.4rem 1.7rem; border:1px solid #DDE6FA; border-radius:22px;
      background:radial-gradient(circle at 92% 18%,#D9E5FF 0,#F6F9FF 33%,#fff 70%); margin-bottom:1.4rem; }
    .eyebrow { color:var(--blue); text-transform:uppercase; letter-spacing:.14em; font-size:11px; font-weight:700; }
    .hero h1 { font-size:35px; margin:.35rem 0 .45rem; }
    .hero p { color:var(--muted); max-width:720px; margin:0; font-size:15px; }
    .card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.2rem; }
    .section-label { color:#344054; font-size:12px; text-transform:uppercase; font-weight:700; letter-spacing:.1em; margin:1.1rem 0 .55rem; }
    .kpi { background:white; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; }
    .kpi-label { color:var(--muted); font-size:12px; }
    .kpi-value { font:700 24px 'Space Grotesk'; margin-top:5px; }
    .hint { color:var(--muted); font-size:13px; line-height:1.5; }
    .stButton > button { border-radius:10px; font-weight:600; border:1px solid #D7DEEC; }
    .stButton > button[kind="primary"] { background:#356AE6; border-color:#356AE6; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _headers() -> dict[str, str]:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _request(method: str, path: str, **kwargs: Any) -> Optional[requests.Response]:
    try:
        with st.spinner("Working..."):
            response = requests.request(
                method, f"{API_BASE_URL}{path}", headers=_headers(), timeout=90, **kwargs
            )
        if response.status_code == 401:
            st.warning("Your session has expired. Sign in again from the sidebar.")
        return response
    except requests.RequestException as exc:
        st.error(f"Unable to reach the API at {API_BASE_URL}: {exc}")
        return None


def _metric(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="hint">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def _format_price(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Price unavailable"
    return f"₹{float(value):,.0f}"


def _friendly_feature_name(feature: str) -> str:
    descriptions = {
        "price_scaled": "listed price",
        "num_reviews_scaled": "number of customer ratings and reviews",
        "avg_rating_scaled": "customer rating",
        "review_avg_rating": "customer review feedback",
        "review_count": "number of written reviews",
        "category_encoded": "category-level pattern",
        "brand_encoded": "brand-level pattern",
        "popularity_percentile": "product popularity in this catalog",
        "interaction_count": "shopper activity recorded for this product",
        "total_events": "category activity in this dataset",
        "launch_days_ago_scaled": "how recently the product was listed",
        "graph_degree": "connections to other catalog items",
        "graph_pagerank": "importance in the product catalog",
    }
    if feature in descriptions:
        return descriptions[feature]
    if feature.startswith("product_text_"):
        return "product name and specification terms"
    if feature.startswith("kg_emb_") or feature.startswith("review_emb_"):
        return "product and review patterns"
    if feature.startswith("event_share_"):
        return "category-level shopping patterns"
    return feature.replace("_scaled", "").replace("_", " ")


def login_sidebar() -> None:
    st.sidebar.markdown(
        '<div class="brand"><span class="brand-mark">✦</span><span class="brand-name">Rankwise</span>'
        '<div class="brand-sub">Explainable product intelligence</div></div>',
        unsafe_allow_html=True,
    )
    if st.session_state.get("access_token"):
        st.sidebar.success("API session active")
        if st.sidebar.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    st.sidebar.caption(f"Backend: {API_BASE_URL}")


def render_auth_page() -> None:
    """Full-page authentication gate shown before any project workspace."""
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown(
            '<div class="hero"><div class="eyebrow">Rankwise workspace</div>'
            '<h1>Understand every product decision.</h1>'
            '<p>Sign in to explore Flipkart product rankings, compare products, '
            'and inspect the evidence behind each ranking.</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card"><b>Built for clear decisions</b><p class="hint">'
            'Your workspace combines category-based ranking, product evidence, and explainable '
            'model signals in one focused interface.</p></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("## Welcome")
        st.caption("Sign in to continue, or create a viewer account.")
        sign_in, sign_up = st.tabs(["Sign in", "Create account"])
        with sign_in:
            username = st.text_input("Username", key="auth_signin_username")
            password = st.text_input("Password", type="password", key="auth_signin_password")
            if st.button("Sign in to workspace", type="primary", use_container_width=True):
                response = _request("POST", "/auth/token", data={"username": username, "password": password})
                if response is not None and response.status_code == 200:
                    st.session_state["access_token"] = response.json()["access_token"]
                    st.rerun()
                elif response is not None:
                    st.error(response.json().get("detail", "Sign in failed."))
        with sign_up:
            username = st.text_input("Choose a username", key="auth_signup_username")
            password = st.text_input("Create a password", type="password", key="auth_signup_password")
            confirm = st.text_input("Confirm password", type="password", key="auth_signup_confirm")
            if st.button("Create viewer account", use_container_width=True):
                if not username.strip() or len(password) < 6:
                    st.error("Enter a username and a password with at least 6 characters.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    response = _request(
                        "POST", "/auth/register",
                        json={"username": username, "password": password, "role": "viewer"},
                    )
                    if response is not None and response.status_code in (200, 201):
                        st.success("Account created. Open Sign in to continue.")
                    elif response is not None:
                        st.error(response.json().get("detail", "Registration failed."))


def render_overview() -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Decision intelligence</div>'
        '<h1>Make every ranking explainable.</h1>'
        '<p>Explore category-based product rankings, compare ranking decisions, and inspect the evidence behind every score.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">How Rankwise works</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="card"><b>01 &nbsp; Discover</b><p class="hint">Search a catalog using product category, brand, or free-text signals.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><b>02 &nbsp; Rank</b><p class="hint">Score products in a category with the hybrid ranking ensemble.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="card"><b>03 &nbsp; Understand</b><p class="hint">Compare products with SHAP contributions and graph evidence.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Getting started</div>', unsafe_allow_html=True)
    if st.session_state.get("access_token"):
        st.success("You are connected. Start with Discover or Rankings in the workspace menu above.")
    else:
        st.info("Start here: create an account or sign in from the sidebar, then choose Rankings above.")


def render_search() -> None:
    st.markdown("## Catalog discovery")
    st.caption(
        "Choose a category here to carry the same selection through Rankings, "
        "Explainability, and category-specific Evaluation. Search terms can be any "
        "product name, brand, or detail within that category."
    )
    category_options = {
        "Laptops": "laptops",
        "Mobiles": "mobiles",
        "Earphones": "earphones",
    }
    if st.session_state.get("discover_category") not in category_options:
        saved_category = st.session_state.get("selected_category")
        st.session_state["discover_category"] = next(
            (
                label for label, category in category_options.items()
                if category == saved_category
            ),
            "Mobiles",
        )
    selected_label = st.selectbox(
        "Category used throughout the workflow",
        list(category_options),
        key="discover_category",
        help="This category is shared by Discover, Rankings, Explainability, and Evaluation.",
    )
    selected_category = category_options[selected_label]
    previous_category = st.session_state.get("selected_category")
    if selected_category != previous_category:
        for state_key in (
            "ranking_df",
            "last_ranked_products",
            "last_user_id",
            "explanation",
            "discover_results",
            "discover_query",
            "discover_product_ids",
            "selected_brand",
        ):
            st.session_state.pop(state_key, None)
    st.session_state["selected_category"] = selected_category

    with st.form("search_form"):
        query = st.text_input(
            "Search products",
            placeholder="Type any product name, brand, or product detail",
            help=f"Search text is unrestricted and will be searched within {selected_label}.",
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            st.caption(f"Selected category: **{selected_label}**")
        with col2:
            top_k = st.slider("Results", 5, 500, 50)
        submitted = st.form_submit_button("Search catalog", type="primary")
    if submitted:
        response = _request(
            "POST",
            "/search/",
            json={
                "query": query,
                "category": selected_category,
                "top_k": top_k,
            },
        )
        if response is None:
            return
        if response.status_code != 200:
            st.error(response.text)
            return
        st.session_state["discover_results"] = response.json()
        st.session_state["discover_query"] = query
        results = response.json()
        result_brands = {
            str(item.get("brand") or "").strip().lower()
            for item in results
            if str(item.get("brand") or "").strip()
        }
        st.session_state["selected_brand"] = (
            next(iter(result_brands)) if len(result_brands) == 1 else None
        )
        st.session_state["discover_product_ids"] = [
            item["product_id"] for item in results
        ] if query.strip() else None
    results = st.session_state.get("discover_results")
    if results is not None:
        st.markdown(
            f"**{len(results)} products found in {selected_label}**"
            + (
                f" for “{st.session_state['discover_query']}”"
                if st.session_state.get("discover_query") else ""
            )
        )
        if results:
            selected_brand = st.session_state.get("selected_brand")
            if selected_brand:
                st.caption(
                    f"Brand filter carried forward: **{selected_brand}**. "
                    "Ranking and brand-specific evaluation will use this brand."
                )
            elif st.session_state.get("discover_query", "").strip():
                st.caption(
                    "This search matched multiple brands; the matching products are "
                    "carried into Rankings, while evaluation remains category-wide."
                )
            st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
        else:
            st.info("No products matched that search in the selected category.")


def render_rankings() -> None:
    st.markdown("## Category ranking")
    selected_category = st.session_state.get("selected_category")
    if not selected_category:
        st.info("Choose a category in Discover first. Rankings will use that same selection.")
        return
    st.caption(
        f"Selected category: **{selected_category}**. This selection is shared with "
        "Discover, Explainability, and Evaluation."
    )
    selected_brand = st.session_state.get("selected_brand")
    if selected_brand:
        st.caption(f"Selected search brand: **{selected_brand}**")
    candidate_product_ids = st.session_state.get("discover_product_ids")
    with st.form("ranking_form"):
        top_k = st.slider("Products to return", 5, 30, 10, key="rank_topk")
        submitted = st.form_submit_button("Generate ranking", type="primary")
    if submitted:
        response = _request(
            "POST",
            "/rank/",
            json={
                "user_id": selected_category,
                "candidate_product_ids": candidate_product_ids,
                "top_k": top_k,
            },
        )
        if response is None:
            return
        if response.status_code != 200:
            st.error(response.text)
            return
        ranked = response.json()["ranked_products"]
        if not ranked:
            st.warning(
                f"No products were found for the '{selected_category}' category. "
                "Check the category in Discover and try again."
            )
            for state_key in (
                "ranking_df",
                "last_ranked_products",
                "last_user_id",
                "explanation",
            ):
                st.session_state.pop(state_key, None)
            return
        st.session_state.pop("explanation", None)
        st.session_state.update(
            last_ranked_products=[item["product_id"] for item in ranked],
            last_user_id=selected_category,
            ranking_df=pd.DataFrame(ranked),
        )
    df = st.session_state.get("ranking_df", pd.DataFrame())
    if df.empty:
        st.info("Generate a ranking for the category selected in Discover.")
        return
    x, y, z = st.columns(3)
    with x: _metric("Products ranked", str(len(df)), "Returned shortlist")
    with y: _metric("Top-ranked item", str(df.iloc[0].get("title") or df.iloc[0]["product_id"]))
    with z: _metric("Top item price", _format_price(df.iloc[0].get("price")))
    st.markdown('<div class="section-label">Ranking results</div>', unsafe_allow_html=True)
    chart = go.Figure(go.Bar(
        x=df["rank_score"],
        y=df.get("title", df["product_id"]),
        orientation="h",
        marker_color="#5E8BFF",
        customdata=df["product_id"],
        hovertemplate="%{y}<br>Model score: %{x:.3f}<br>Product ID: %{customdata}<extra></extra>",
    ))
    chart.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Relative model score (higher ranks first)", yaxis_title="",
    )
    st.plotly_chart(chart, use_container_width=True)
    display_df = df.copy()
    display_df["price"] = display_df["price"].map(_format_price)
    display_df = display_df.rename(columns={
        "rank_position": "Rank",
        "title": "Product",
        "price": "Price",
        "brand": "Brand",
        "num_reviews": "Review count",
        "rank_score": "Relative model score",
        "product_id": "Product ID",
    })
    preferred_columns = [
        name for name in [
            "Rank", "Product", "Price", "Brand", "Review count",
            "Relative model score", "Product ID",
        ] if name in display_df.columns
    ]
    st.dataframe(
        display_df[preferred_columns], hide_index=True, use_container_width=True
    )
    st.caption(
        "Rankings use product title/specification text, category, brand and review "
        "volume; price is displayed for your comparison but does not affect rank. "
        "Ratings supply the training relevance label, not a model input. Scores order "
        "items within this category and are not probabilities or quality guarantees."
        " Select a product in Explainability for a plain-language summary."
    )


def render_explanations() -> None:
    st.markdown("## Explain a ranking decision")
    st.caption("Select two products to see exactly which model signals created the ranking gap.")
    selected_category = st.session_state.get("selected_category")
    if not selected_category:
        st.info("Choose a category in Discover first. Explainability uses that same selection.")
        return
    candidates = st.session_state.get("last_ranked_products", [])
    if not candidates:
        st.warning(
            f"Generate a ranking for '{selected_category}' first. "
            "Your ranked products will appear here for comparison."
        )
        return
    if st.session_state.get("last_user_id") != selected_category:
        st.warning("The category changed. Generate a new ranking before explaining products.")
        return
    st.caption(f"Selected category: **{selected_category}**")
    ranking_df = st.session_state.get("ranking_df", pd.DataFrame())
    title_by_id = (
        ranking_df.set_index("product_id")["title"].to_dict()
        if not ranking_df.empty and "title" in ranking_df.columns else {}
    )
    price_by_id = (
        ranking_df.set_index("product_id")["price"].to_dict()
        if not ranking_df.empty and "price" in ranking_df.columns else {}
    )

    def product_label(product_id: str) -> str:
        title = title_by_id.get(product_id, product_id)
        return f"{title} — {_format_price(price_by_id.get(product_id))}"

    first, second = st.columns(2)
    with first:
        product_id = st.selectbox(
            "Selected product", candidates, format_func=product_label
        )
    with second:
        compare_options = ["(none)"] + [item for item in candidates if item != product_id]
        compare_to = st.selectbox(
            "Compare against",
            compare_options,
            format_func=lambda item: "(none)" if item == "(none)" else product_label(item),
        )
    if st.button("Explain decision", type="primary", use_container_width=True):
        st.session_state.pop("explanation", None)
        response = _request(
            "POST", "/explain/",
            json={"user_id": selected_category, "product_id": product_id,
                  "compare_to_product_id": None if compare_to == "(none)" else compare_to},
        )
        if response is None:
            return
        if response.status_code != 200:
            st.error(response.text)
            return
        st.session_state["explanation"] = response.json()
    data = st.session_state.get("explanation")
    if not data:
        st.info("Choose a product and click Explain decision.")
        return
    selected_title = data.get("product_title") or product_label(data["product_id"])
    st.markdown("### " + selected_title)
    st.caption(
        f"Price: {_format_price(data.get('price'))}"
        + (
            f" · Compared with {data.get('compared_product_title')} "
            f"({_format_price(data.get('compared_price'))})"
            if data.get("compared_product_id") else ""
        )
    )
    st.info(data.get("plain_language_summary", data["explanation_text"]))
    with st.expander("Model explanation details"):
        st.write(data["explanation_text"])
    if data.get("compared_product_id"):
        if data["score_difference"] >= 0:
            comparison_heading = (
                f'{selected_title} ranked above '
                f'{data.get("compared_product_title") or product_label(data["compared_product_id"])}'
            )
        else:
            comparison_heading = (
                f'{data.get("compared_product_title") or product_label(data["compared_product_id"])} '
                f'ranked above {selected_title}'
            )
        st.markdown(f"#### {comparison_heading}")
        a, b, c = st.columns(3)
        with a: _metric("Selected score", f'{data["rank_score"]:.4f}')
        with b: _metric("Compared score", f'{data["compared_rank_score"]:.4f}')
        with c: _metric("Score advantage", f'{data["score_difference"]:+.4f}')
        comparison = pd.DataFrame(data.get("comparison_features", []))
        if not comparison.empty:
            comparison["favors"] = comparison["favors"].map(
                {
                    "product_a": selected_title,
                    "product_b": data.get("compared_product_title")
                    or product_label(data["compared_product_id"]),
                }
            )
            comparison = comparison.rename(columns={"delta_shap": "contribution difference"})
            comparison["feature"] = comparison["feature"].map(_friendly_feature_name)
            st.markdown('<div class="section-label">What created the difference?</div>', unsafe_allow_html=True)
            st.dataframe(comparison[["feature", "contribution difference", "favors"]], hide_index=True, use_container_width=True)
    else:
        st.markdown("#### What influenced this ranking?")
    a, b, c = st.columns(3)
    with a: _metric("Confidence", f'{data["confidence_score"]:.2f}', "Grounding confidence")
    with b: _metric("ECS", f'{data["ecs_score"]:.2f}', "Explainability confidence")
    with c: _metric("Verification", "Passed" if data["passed_verification"] else "Review", "Evidence check")
    features = pd.DataFrame(data.get("top_shap_features", []))
    if not features.empty:
        features["feature"] = features["feature"].map(_friendly_feature_name)
        st.markdown('<div class="section-label">Feature contributions</div>', unsafe_allow_html=True)
        chart = go.Figure(go.Bar(
            x=features["shap_value"], y=features["feature"], orientation="h",
            marker_color=["#E25563" if value < 0 else "#26A69A" for value in features["shap_value"]],
        ))
        chart.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Contribution to rank")
        st.plotly_chart(chart, use_container_width=True)
    render_graph_path(data.get("graph_path", []))


def render_graph_path(path: list) -> None:
    st.markdown('<div class="section-label">Knowledge graph evidence</div>', unsafe_allow_html=True)
    if not path:
        st.info("No direct knowledge-graph path was found for this decision.")
        return
    graph = nx.DiGraph()
    for edge in path:
        graph.add_edge(edge["from"], edge["to"], relation=edge["relation"])
    pos = nx.spring_layout(graph, seed=42)
    edge_x, edge_y = [], []
    for source, target in graph.edges():
        edge_x += [pos[source][0], pos[target][0], None]
        edge_y += [pos[source][1], pos[target][1], None]
    figure = go.Figure([
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#B7C2D8")),
        go.Scatter(
            x=[pos[node][0] for node in graph.nodes()], y=[pos[node][1] for node in graph.nodes()],
            mode="markers+text", text=list(graph.nodes()), textposition="top center",
            marker=dict(size=17, color="#356AE6"),
        ),
    ])
    figure.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(figure, use_container_width=True)


def render_evaluation() -> None:
    st.markdown("## Model evaluation")
    st.caption("Compare dataset sizes, held-out ranking quality, and inference efficiency.")
    st.caption(
        "Popularity uses training-split interaction counts; unseen products tie. "
        "Ranking metrics average tied scores instead of using file order. ROC-AUC is "
        "discrimination (0.5 = chance), not classification accuracy."
    )
    selected_category = st.session_state.get("selected_category")
    if selected_category:
        selected_brand = st.session_state.get("selected_brand")
        scope_label = (
            f"{selected_category} / {selected_brand}"
            if selected_brand else selected_category
        )
        st.markdown(f"### Evaluation for selected category: {scope_label}")
    else:
        st.info("Choose a category in Discover to see its category-specific evaluation.")
    datasets_response = _request("GET", "/evaluation/datasets")
    if datasets_response is not None and datasets_response.status_code == 200:
        dataset_report = datasets_response.json()
        flipkart = next(
            (
                dataset for dataset in dataset_report.get("datasets", [])
                if dataset.get("dataset_id") == "flipkart"
                and dataset.get("status") == "evaluated"
            ),
            None,
        )
        category_result = (
            flipkart.get("category_metrics", {}).get(selected_category)
            if flipkart and selected_category else None
        )
        category_baseline = (
            flipkart.get("category_baseline_metrics", {}).get(selected_category)
            if flipkart and selected_category else None
        )
        brand_metrics = (
            flipkart.get("category_brand_metrics", {})
            .get(selected_category, {}).get(selected_brand)
            if flipkart and selected_category and selected_brand else None
        )
        brand_baseline = (
            flipkart.get("category_brand_baseline_metrics", {})
            .get(selected_category, {}).get(selected_brand)
            if flipkart and selected_category and selected_brand else None
        )
        if category_result:
            evaluation_result = brand_metrics or category_result
            evaluation_baseline = brand_baseline or category_baseline
            evaluation_metrics = evaluation_result["metrics"]
            baseline_metrics = (
                evaluation_baseline.get("metrics", {})
                if evaluation_baseline else {}
            )
            st.caption(
                f"Held-out comparison for {evaluation_result['products']:,} products "
                f"({evaluation_result['test_rows']:,} test rows) in the selected "
                f"{'brand and category' if brand_metrics else 'category'}."
            )
            metric_rows = []
            metric_columns = st.columns(5)
            for column, metric_label, metric_key in zip(
                metric_columns,
                ("NDCG@10", "MAP", "MRR", "Precision@10", "ROC-AUC"),
                ("ndcg_at_10", "map", "mrr", "precision_at_10", "roc_auc"),
            ):
                value = evaluation_metrics.get(metric_key)
                baseline_value = baseline_metrics.get(metric_key)
                with column:
                    _metric(
                        metric_label,
                        "N/A" if value is None else repr(float(value)),
                        (
                            "Popularity baseline: N/A"
                            if baseline_value is None
                            else f"Popularity baseline: {repr(float(baseline_value))}"
                        ),
                    )
                metric_rows.append({
                    "Metric": metric_label,
                    "Hybrid ensemble": value,
                    "Popularity baseline": baseline_value,
                })
            st.dataframe(
                pd.DataFrame([
                    {
                        "Metric": row["Metric"],
                        "Hybrid ensemble": (
                            "N/A" if row["Hybrid ensemble"] is None
                            else repr(float(row["Hybrid ensemble"]))
                        ),
                        "Popularity baseline": (
                            "N/A" if row["Popularity baseline"] is None
                            else repr(float(row["Popularity baseline"]))
                        ),
                    }
                    for row in metric_rows
                ]),
                hide_index=True,
                use_container_width=True,
            )
        elif selected_category:
            st.warning(
                f"No held-out metrics were found for '{selected_category}'. "
                "Choose a category available in the evaluated Flipkart dataset."
            )

        if category_result:
            st.markdown("### Ranking quality")
            chart_data = pd.DataFrame(metric_rows)
            figure = go.Figure()
            figure.add_bar(
                name="Hybrid ensemble",
                x=chart_data["Metric"],
                y=chart_data["Hybrid ensemble"],
                marker_color="#356AE6",
                hovertemplate="%{x}<br>Hybrid: %{y:.17g}<extra></extra>",
            )
            figure.add_bar(
                name="Popularity baseline",
                x=chart_data["Metric"],
                y=chart_data["Popularity baseline"],
                marker_color="#AAB6CC",
                hovertemplate="%{x}<br>Popularity baseline: %{y:.17g}<extra></extra>",
            )
            figure.update_layout(
                barmode="group",
                height=370,
                yaxis=dict(title="Score", range=[0, 1]),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(figure, use_container_width=True)

        st.markdown("### Overall dataset comparison")
        st.caption(
            "These rows summarize each entire dataset. The category-specific metrics above "
            "use the category selected in Discover."
        )
        rows = []
        for dataset in dataset_report.get("datasets", []):
            split = dataset.get("split", {})
            metrics = dataset.get("metrics", {})
            efficiency = dataset.get("efficiency", {})
            csv_lines = dataset.get("csv_data_lines", {})
            physical_csv_lines = csv_lines.get(
                "interactions.csv", csv_lines.get("flipkart_combined.csv")
            )
            rows.append({
                "Dataset": dataset["name"],
                "Status": dataset["status"].replace("_", " ").title(),
                "Raw size (MB)": dataset.get("raw_size_mb"),
                "Training groups": split.get("user_groups"),
                "Products in splits": split.get("products_in_train"),
                "CSV physical lines": physical_csv_lines,
                "Train rows": split.get("train_rows"),
                "Validation rows": split.get("validation_rows"),
                "Test rows": split.get("test_rows"),
                "NDCG@10": metrics.get("ndcg_at_10"),
                "MAP": metrics.get("map"),
                "MRR": metrics.get("mrr"),
                "Baseline MRR": dataset.get("baseline_metrics", {}).get("mrr"),
                "Precision@10": metrics.get("precision_at_10"),
                "ROC-AUC": metrics.get("roc_auc"),
                "Inference (sec/100 rows)": efficiency.get("latency_seconds_per_100_rows"),
                "Peak traced memory (MB)": efficiency.get("peak_memory_mb"),
                "Train time (sec)": dataset.get("training_seconds"),
                "Notes": dataset.get("note"),
            })
        if rows:
            st.markdown("### Dataset-by-dataset results")
            chart_rows = [
                row for row in rows
                if row["Status"] == "Evaluated" and row["NDCG@10"] is not None
            ]
            if chart_rows:
                st.markdown("#### Dataset ranking metric comparison")
                chart_frame = pd.DataFrame(chart_rows)
                dataset_figure = go.Figure()
                dataset_figure.add_bar(
                    name="Hybrid NDCG@10",
                    x=chart_frame["Dataset"],
                    y=chart_frame["NDCG@10"],
                    marker_color="#356AE6",
                )
                dataset_figure.add_bar(
                    name="Popularity NDCG@10",
                    x=chart_frame["Dataset"],
                    y=[
                        next(
                            (
                                dataset.get("baseline_metrics", {}).get("ndcg_at_10")
                                for dataset in dataset_report.get("datasets", [])
                                if dataset["name"] == dataset_name
                            ),
                            None,
                        )
                        for dataset_name in chart_frame["Dataset"]
                    ],
                    marker_color="#AAB6CC",
                )
                dataset_figure.update_layout(
                    barmode="group",
                    height=360,
                    yaxis=dict(title="NDCG@10", range=[0, 1]),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(dataset_figure, use_container_width=True)
            dataset_frame = pd.DataFrame(rows)
            for metric_column in (
                "NDCG@10", "MAP", "MRR", "Baseline MRR", "Precision@10", "ROC-AUC",
            ):
                if metric_column in dataset_frame:
                    dataset_frame[metric_column] = dataset_frame[metric_column].map(
                        lambda value: (
                            "N/A" if value is None or pd.isna(value)
                            else repr(float(value))
                        )
                    )
            st.dataframe(
                dataset_frame, hide_index=True, use_container_width=True
            )
            st.caption(dataset_report.get("row_count_note", ""))
            with st.expander("Dataset sizes by source file"):
                for dataset in dataset_report.get("datasets", []):
                    with st.container():
                        st.markdown(f"**{dataset['name']}** — {dataset['status'].replace('_', ' ').title()}")
                        size_rows = [
                            {"File": name, "CSV data lines": count}
                            for name, count in dataset.get("csv_data_lines", {}).items()
                        ]
                        if size_rows:
                            st.dataframe(
                                pd.DataFrame(size_rows),
                                hide_index=True,
                                use_container_width=True,
                            )
                        st.caption(
                            f"Total source size: {dataset.get('raw_size_mb', 0):,.2f} MB. "
                            f"{dataset.get('note', '')}"
                        )
    elif datasets_response is not None:
        st.info(
            "Per-dataset results are not generated yet. Run "
            "`python -m evaluation.evaluate_datasets` from the project directory."
        )



def main() -> None:
    if not st.session_state.get("access_token"):
        render_auth_page()
        return
    login_sidebar()
    st.markdown("### Product ranking command center")
    page = st.radio(
        "Workspace",
        ["Overview", "Discover", "Rankings", "Explainability", "Evaluation"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if page == "Overview":
        render_overview()
    elif page == "Discover":
        render_search()
    elif page == "Rankings":
        render_rankings()
    elif page == "Explainability":
        render_explanations()
    else:
        render_evaluation()


if __name__ == "__main__":
    main()
