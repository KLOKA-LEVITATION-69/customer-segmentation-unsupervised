"""Interactive explorer for the unsupervised customer segmentation project.

Run:  streamlit run src/app.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import (AGE_COL, FIG_DIR, INCOME_COL, MODEL_DIR, REPORT_DIR,
                    SPEND_COL)

st.set_page_config(page_title="Customer Segmentation - Unsupervised Learning",
                   layout="wide", page_icon="🛍️")

RAW_COLS = {"Age": AGE_COL, "Income (k$)": INCOME_COL, "Spending score": SPEND_COL}


@st.cache_data
def load_data():
    from data_utils import get_clean_data
    return get_clean_data()


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_DIR / "kmeans_pipeline.joblib")


df = load_data()
st.title("🛍️ Unsupervised Customer Segmentation")
st.caption("Mall Customers dataset · K-Means · Hierarchical · GMM · DBSCAN · PCA · t-SNE")

tab_explore, tab_cluster, tab_segments, tab_predict = st.tabs(
    ["📊 Explore", "🧠 Cluster live", "👥 Segments", "🔮 Predict a customer"])


def metrics_block(Xs, labels):
    n = len(set(labels)) - (1 if -1 in labels else 0)
    if n < 2:
        st.warning("This configuration produced fewer than 2 clusters.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters found", n)
    c2.metric("Silhouette (↑)", f"{silhouette_score(Xs, labels):.3f}")
    c3.metric("Davies-Bouldin (↓)", f"{davies_bouldin_score(Xs, labels):.3f}")
    c4.metric("Calinski-Harabasz (↑)", f"{calinski_harabasz_score(Xs, labels):.0f}")


# ------------------------------------------------------------------ Explore
with tab_explore:
    c1, c2, c3 = st.columns(3)
    x_col = c1.selectbox("X axis", list(RAW_COLS), 1)
    y_col = c2.selectbox("Y axis", list(RAW_COLS), 2)
    color = c3.selectbox("Colour by", ["Gender", "AgeBand", "IncomeBand"], 0)

    fig = px.scatter(df, x=RAW_COLS[x_col], y=RAW_COLS[y_col], color=color,
                     hover_data=[AGE_COL, INCOME_COL, SPEND_COL],
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title=f"{y_col} vs {x_col}")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.dataframe(df.describe().round(2), use_container_width=True)
    with right:
        st.subheader("Quick stats")
        st.markdown(
            f"- **{len(df)}** customers, no labels - the clusters are *discovered*, not given\n"
            f"- Gender: {df['Gender'].value_counts().to_dict()}\n"
            f"- Mean age **{df[AGE_COL].mean():.1f}** · income **{df[INCOME_COL].mean():.1f}k$** "
            f"· spending **{df[SPEND_COL].mean():.1f}**\n"
            f"- Income↔spending correlation is only "
            f"**{df[[INCOME_COL, SPEND_COL]].corr().iloc[0, 1]:.2f}** → income alone "
            f"does not explain spending (why clustering beats intuition)")

# ------------------------------------------------------------- Cluster live
with tab_cluster:
    st.markdown("Train a clustering model **right now** on the features you choose.")
    feat = st.multiselect("Features", ["Age", INCOME_COL, SPEND_COL],
                          default=[INCOME_COL, SPEND_COL])
    algo = st.radio("Algorithm", ["K-Means", "Hierarchical (Ward)", "GMM", "DBSCAN"],
                    horizontal=True)

    if len(feat) < 2:
        st.info("Pick at least 2 features (distance-based clustering needs a plane).")
    else:
        X = df[feat]
        Xs = StandardScaler().fit_transform(X)

        labels = None
        if algo == "K-Means":
            k = st.slider("k", 2, 10, 5)
            labels = KMeans(n_clusters=k, n_init=25, random_state=42).fit_predict(Xs)
        elif algo == "Hierarchical (Ward)":
            k = st.slider("k", 2, 10, 5)
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs)
        elif algo == "GMM":
            k = st.slider("k", 2, 10, 5)
            labels = GaussianMixture(n_components=k, n_init=10, random_state=42).fit_predict(Xs)
        else:
            eps = st.slider("DBSCAN eps (standardised units)", 0.1, 1.5, 0.4, 0.05)
            min_samples = st.slider("min_samples", 2, 12, 5)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)

        metrics_block(Xs, labels)

        px_x = st.selectbox("Plot X", feat, 0)
        px_y = st.selectbox("Plot Y", [f for f in feat if f != px_x], 0)
        plot_df = df.copy()
        plot_df["cluster"] = labels
        fig = px.scatter(plot_df, x=px_x, y=px_y, color=plot_df["cluster"].astype(str),
                         hover_data=[AGE_COL, INCOME_COL, SPEND_COL],
                         title=f"{algo} result (−1 = noise for DBSCAN)")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Segments
with tab_segments:
    st.markdown("The **saved K-Means (k=5) model** - the deliverable from `src/clustering.py`.")
    scored_path = REPORT_DIR / "scored_customers.csv"
    if scored_path.exists():
        scored = pd.read_csv(scored_path)
        metrics = json.loads((REPORT_DIR / "model_comparison.json").read_text())
        prof = pd.DataFrame(metrics["kmeans_profile"]).rename(columns={
            "n": "customers", "age_mean": "avg age", "income_mean": "avg income (k$)",
            "spend_mean": "avg spending", "male_pct": "% male"})
        st.dataframe(prof, use_container_width=True, hide_index=True)

        fig = px.scatter_matrix(scored[["Cluster_KMeans", AGE_COL, INCOME_COL, SPEND_COL]],
                                dimensions=[AGE_COL, INCOME_COL, SPEND_COL],
                                color=scored["Cluster_KMeans"].astype(str), height=650)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Static report figures")
        for f in sorted(FIG_DIR.glob("*.png")):
            st.image(str(f), caption=f.stem)
    else:
        st.info("Run `python src/clustering.py` first to generate the saved segments.")

# ----------------------------------------------------------------- Predict
with tab_predict:
    st.markdown("Assign a **new customer** to a segment with the saved K-Means pipeline "
                "(trained on the income × spending plane).")
    pipe = load_pipeline()
    model_feats = pipe["features"]
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age (context only)", 18, 90, 30,
                          help="The saved model clusters on income and spending; "
                               "age is shown for segment context.")
    inc = c2.number_input("Annual income (k$)", 10, 200, 60)
    spend = c3.slider("Spending score (1-100)", 1, 99, 55)

    values = {AGE_COL: age, INCOME_COL: inc, SPEND_COL: spend}
    Xnew = pd.DataFrame([[values[f] for f in model_feats]], columns=model_feats)
    cluster = int(pipe["model"].predict(pipe["scaler"].transform(Xnew))[0])

    prof = pd.read_csv(REPORT_DIR / "cluster_profiles.csv")
    row = prof[prof["Cluster_KMeans"] == cluster].iloc[0]
    st.success(
        f"This customer lands in **Segment {cluster}** — "
        f"avg profile of that segment: age **{row['age_mean']:.0f}**, income "
        f"**{row['income_mean']:.0f}k$**, spending **{row['spend_mean']:.0f}**, "
        f"**{row['n']}** customers in it.")

st.divider()
st.caption("Major project · Unsupervised Learning · unlabeled data, discovered structure.")
