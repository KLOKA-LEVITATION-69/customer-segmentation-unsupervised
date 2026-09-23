# Customer Segmentation — Unsupervised Learning (Major Project)

An end-to-end **unsupervised machine learning** project: four clustering
algorithms discover natural customer segments in a shopping-mall dataset
**without any labels**, validated with internal indices, and deployed in an
interactive Streamlit app.

> Brief (from the project DOCX): *Unsupervised learning trains on unlabeled
> data; the model identifies patterns, structures and relationships without
> prior knowledge of labels or outcomes. As a team, choose your own model and
> dataset and prepare an abstract.*

- **Model choice:** K-Means · Agglomerative Hierarchical (Ward) · GMM · DBSCAN,
  with PCA and t-SNE for structure visualisation.
- **Dataset choice:** Mall Customers — 200 mall visitors, 4 attributes, **no
  target column** (see `data/Mall_Customers.csv`).
- **Abstract:** [`ABSTRACT.md`](ABSTRACT.md).

## Project structure
```
customer-segmentation-unsupervised/
├── data/Mall_Customers.csv          # the unlabeled dataset (200 rows)
├── src/
│   ├── config.py                    # paths + feature definitions
│   ├── data_utils.py                # loading, cleaning, feature engineering
│   ├── eda.py                       # 8 EDA figures + insights report
│   ├── clustering.py                # 4 algorithms + validation + projections
│   └── app.py                       # Streamlit explorer
├── outputs/
│   ├── figures/                     # fig01 ... fig17 PNGs
│   ├── models/kmeans_pipeline.joblib# saved model + scaler
│   └── reports/                     # metrics, profiles, scored customers
├── ABSTRACT.md · README.md · requirements.txt · .gitignore
```

## How to run
```bash
pip install -r requirements.txt
python src/eda.py          # figures + eda_summary.txt          (~10 s)
python src/clustering.py   # train + compare 4 models          (~1 min)
streamlit run src/app.py   # interactive dashboard
```

## Results

**Primary space: Annual Income × Spending Score (standardised), k = 5**

| Model | Silhouette (↑) | Davies–Bouldin (↓) | Calinski–Harabasz (↑) |
|---|---|---|---|
| **K-Means** | **0.555** | **0.572** | **248.6** |
| GMM (full covariance) | 0.554 | 0.576 | 244.9 |
| Hierarchical (Ward) | 0.554 | 0.578 | 244.4 |
| DBSCAN (eps=0.35, min_samples=4) | 0.426 | 1.413 | 81.0 |

DBSCAN (tuned via k-distance grid search) finds **7 dense regions** and flags
**8 % of customers as noise** — the transitional shoppers between blobs.

**Extended space (adding Age):** silhouette drops to 0.417 (K-Means k=5) —
age blurs the plane's clean structure, but the segments become more actionable
(the high-spending groups are visibly younger).

**Choosing k (2…10):** silhouette peaks and Davies–Bouldin bottoms out exactly
at k = 5; the elbow in inertia agrees (`fig09_k_selection.png`).

### Discovered segments (K-Means, k = 5, exact profiles)

| Segment | n | Avg age | Avg income | Avg spending | Interpretation → action |
|---|---|---|---|---|---|
| 1 | 39 | 32.7 | 86.5 k$ | 82.1 | **VIPs** → premium retention, early access |
| 2 | 22 | 25.3 | 25.7 k$ | 79.4 | **Young enthusiasts** → loyalty app, BNPL offers |
| 0 | 81 | 42.7 | 55.3 k$ | 49.5 | **Moderate centrists** → upsell / cross-sell |
| 4 | 23 | 45.2 | 26.3 k$ | 20.9 | **Budget shoppers** → discounts, bundles |
| 3 | 35 | 41.1 | 88.2 k$ | 17.1 | **Careful spenders** → trust campaigns, personalised ads |

## Key insights
- Income and spending score are essentially **uncorrelated** (r ≈ 0.0), yet
  jointly they form **five clean blobs** — the reason a 2-feature K-Means works
  so well here.
- All three partitioning models (K-Means, Ward, GMM) recover **the same five
  segments** (silhouettes within 0.001 of each other) — strong evidence the
  structure is real, not an artefact of one algorithm.
- PCA on the extended space spreads variance almost evenly (44 % / 33 % / 22 %),
  i.e. no single axis dominates — all three features carry information.
- DBSCAN's noise points are *transitional* customers between segments — a nice
  demonstration of density-based outlier detection.

## Notes for the viva
- Why scale? K-Means and DBSCAN are distance-based; raw income (15–137 k$)
  would dominate spending (1–99).
- Why k = 5? Internal indices + the visibly five-blob structure + marketing
  interpretability (matches the classic "customer value matrix").
- Why internal metrics? There are no labels — external accuracy measures are
  undefined in unsupervised learning.
