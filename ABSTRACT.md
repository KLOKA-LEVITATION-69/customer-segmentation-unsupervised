# ABSTRACT

## Unsupervised Learning: Customer Segmentation for Targeted Marketing

**Model.** This project applies unsupervised learning — machine learning on
*unlabeled* data, where the algorithm must discover structure without any
supervisory signal — to one of its most commercially valuable problems:
market segmentation. Four clustering algorithms are implemented and compared:
**K-Means** (centroid-based), **Agglomerative Hierarchical clustering with
Ward linkage** (connectivity-based), **Gaussian Mixture Models** (probabilistic,
soft assignment), and **DBSCAN** (density-based, with automatic outlier
detection). Because no labels exist, the number of clusters is chosen with
internal validation only: the elbow method on inertia, the **silhouette
coefficient**, the **Davies–Bouldin index** and the **Calinski–Harabasz
index**, computed over k = 2…10. The high-dimensional standardised feature
space (Age, Annual Income, Spending Score) is inspected with **PCA** (variance
decomposition and loadings) and **t-SNE** (non-linear 2-D map), and every
algorithm's solution is judged with the same internal indices.

**Dataset.** The *Mall Customers* dataset — 200 real mall visitors described
by Gender, Age, Annual Income (k$) and a Spending Score (1–100) assigned by the
mall from purchasing behaviour. It contains **no target column**: exactly the
unlabeled setting the brief requires. Pre-processing includes dropping the
identifier, plausibility filtering, duplicate checks, and feature engineering
(age bands, income bands, normalised coordinates and a spending-to-income
ratio). Exploratory analysis (8 figures) shows income and spending are nearly
uncorrelated (r ≈ 0.0) yet the income–spending plane reveals a striking
five-group visual structure — the discovery target for clustering.

**Results.** K-Means with k = 5 achieves the best internal validation of the
four models (silhouette 0.555, Davies–Bouldin 0.572, Calinski–Harabasz 248.6)
and recovers five operationally meaningful segments: **(1) VIPs** (high income,
high spending), **(2) Young enthusiasts** (low income but high spending),
**(3) Moderate centrists**, **(4) Budget shoppers** (low income, low spending)
and **(5) Careful spenders** (high income, low spending). Hierarchical
clustering and GMM independently recover essentially the same partition
(silhouettes within 0.001 — strong evidence the structure is real), while a
k-distance-tuned DBSCAN finds seven dense regions and flags 8 % of customers
as transitional noise. Each segment is translated into a concrete marketing
action (VIP retention, youth loyalty programmes, careful-spenders conversion
campaigns, etc.).

**Deliverables.** Reproducible Python pipeline (`src/`: `config`,
`data_utils`, `eda`, `clustering`, `app`), 15+ figures, metric reports
(`model_comparison.json`, `cluster_profiles.csv`), a serialised K-Means +
scaler pipeline for scoring new customers, and an interactive **Streamlit**
application for live re-clustering, segment profiling and single-customer
segment prediction.

**Keywords:** unsupervised learning, clustering, K-Means, hierarchical
clustering, DBSCAN, Gaussian Mixture Models, silhouette analysis, PCA, t-SNE,
customer segmentation, marketing analytics.
