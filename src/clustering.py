"""Unsupervised learning core: K-Means, Hierarchical, DBSCAN, GMM.

No labels are used anywhere in training - this is pure unsupervised learning.
Cluster "quality" is measured with internal indices (silhouette, Davies-Bouldin,
Calinski-Harabasz); the marketing interpretation happens only afterwards.

Two feature spaces are analysed:
  * PLANE (primary): Annual Income + Spending Score - the classic space where
    the five-blob structure lives.
  * EXTENDED: Age + Income + Spending - tests whether age refines the segments.

Run:  python src/clustering.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score, silhouette_samples)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from config import (FIG_DIR, INCOME_COL, MODEL_DIR, RANDOM_STATE, REPORT_DIR,
                    SPEND_COL)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
PALETTE = sns.color_palette("tab10")

# Internal metrics: silhouette wants HIGH, Davies-Bouldin wants LOW,
# Calinski-Harabasz wants HIGH. k=5 is stated up front as the business prior.
K_PRIOR = 5

PLANE_FEATURES = [INCOME_COL, SPEND_COL]
EXTENDED_FEATURES = ["Age", INCOME_COL, SPEND_COL]


# ---------------------------------------------------------------- helpers
def scale_features(X: pd.DataFrame):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def internal_metrics(X, labels):
    n_labels = len(set(labels)) - (1 if -1 in labels else 0)
    if n_labels < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan, "n_clusters": n_labels}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "n_clusters": n_labels,
    }


# ---------------------------------------------------------------- selection
def k_selection_study(Xs, tag=""):
    """Elbow (inertia) + silhouette + Davies-Bouldin + CH across k=2..10."""
    ks = range(2, 11)
    inertias, sils, dbs, chs = [], [], [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE).fit(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, km.labels_))
        dbs.append(davies_bouldin_score(Xs, km.labels_))
        chs.append(calinski_harabasz_score(Xs, km.labels_))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Choosing k{tag} (higher silhouette/CH better, lower DB better)",
                 fontweight="bold")
    axes[0, 0].plot(list(ks), inertias, "o-")
    axes[0, 0].set_title("Elbow method (inertia)")
    axes[0, 1].plot(list(ks), sils, "o-", color="#2ca02c")
    axes[0, 1].set_title("Silhouette score")
    axes[1, 0].plot(list(ks), dbs, "o-", color="#d62728")
    axes[1, 0].set_title("Davies-Bouldin (lower = better)")
    axes[1, 1].plot(list(ks), chs, "o-", color="#9467bd")
    axes[1, 1].set_title("Calinski-Harabasz")
    for ax in axes.flat:
        ax.set_xlabel("k")
        ax.axvline(K_PRIOR, color="red", ls="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig09_k_selection{tag}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig09_k_selection{tag}.png")

    table = pd.DataFrame({"k": list(ks), "inertia": inertias, "silhouette": sils,
                          "davies_bouldin": dbs, "calinski_harabasz": chs})
    return table


def silhouette_plot(Xs, labels, title, fname):
    values = silhouette_samples(Xs, labels)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y_lower = 10
    for i, c in enumerate(sorted(set(labels))):
        vals = np.sort(values[labels == c])
        size = len(vals)
        y_upper = y_lower + size
        color = "grey" if c == -1 else PALETTE[i % len(PALETTE)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         facecolor=color, edgecolor=color, alpha=0.75)
        ax.text(-0.06, y_lower + 0.5 * size, str(c), ha="right")
        y_lower = y_upper + 8
    ax.axvline(values.mean(), color="red", ls="--",
               label=f"mean silhouette = {values.mean():.3f}")
    ax.set_title(title)
    ax.set_xlabel("Silhouette coefficient")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------- models
def fit_kmeans(Xs, k=K_PRIOR):
    km = KMeans(n_clusters=k, n_init=50, max_iter=500, random_state=RANDOM_STATE)
    labels = km.fit_predict(Xs)
    return km, labels


def fit_hierarchical(Xs, k=K_PRIOR):
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(Xs)
    return model, labels


def fit_gmm(Xs, k=K_PRIOR):
    model = GaussianMixture(n_components=k, covariance_type="full",
                            n_init=20, random_state=RANDOM_STATE)
    model.fit(Xs)
    labels = model.predict(Xs)
    return model, labels


def tune_dbscan(Xs, max_noise=0.10, min_clusters=4):
    """Grid-search eps via k-distance heuristic + min_samples.

    Constraints: at least `min_clusters` dense regions and at most `max_noise`
    fraction of points labelled as noise; among survivors pick the best
    silhouette. DBSCAN is allowed to disagree with K-Means about the number of
    clusters - that is part of the comparison.
    """
    # k-distance graph for the 5th nearest neighbour -> principled eps range.
    nn = NearestNeighbors(n_neighbors=6).fit(Xs)
    dist, _ = nn.kneighbors(Xs)
    kdist = np.sort(dist[:, -1])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(kdist)
    ax.set_title("k-distance graph (5th NN) - elbow suggests eps")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel("5th nearest neighbour distance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_dbscan_kdistance.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved fig11_dbscan_kdistance.png")

    best = None
    for eps in np.round(np.arange(0.20, 0.75, 0.05), 2):
        for min_samples in (4, 5, 6, 8):
            db = DBSCAN(eps=float(eps), min_samples=min_samples).fit(Xs)
            m = internal_metrics(Xs, db.labels_)
            noise = float(np.mean(db.labels_ == -1))
            if (m["n_clusters"] < min_clusters or np.isnan(m["silhouette"])
                    or noise > max_noise):
                continue
            if best is None or m["silhouette"] > best["silhouette"]:
                best = {"eps": float(eps), "min_samples": min_samples,
                        **m, "noise_frac": noise}
    return best


def fit_dbscan(Xs, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(Xs)
    return model, model.labels_


# ---------------------------------------------------------------- projections
def pca_analysis(Xs, feature_names):
    pca = PCA(n_components=len(feature_names), random_state=RANDOM_STATE).fit(Xs)
    explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(range(1, len(explained) + 1), np.cumsum(explained), "o-")
    axes[0].set_title("PCA cumulative explained variance")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Cumulative ratio")
    axes[0].axhline(0.9, color="red", ls="--", alpha=0.5)

    loadings = pca.components_.T
    x = np.arange(len(feature_names))
    for i in range(2):
        axes[1].bar(x + (i - 0.5) * 0.35, loadings[:, i], width=0.35,
                    label=f"PC{i + 1} ({explained[i]:.0%})")
    axes[1].set_xticks(x, feature_names)
    axes[1].set_title("PCA loadings")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_pca.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved fig12_pca.png")
    return pca, explained


def tsne_projection(Xs, labels, title, fname):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                init="pca", random_state=RANDOM_STATE)
    Z = tsne.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for c in sorted(set(labels)):
        mask = labels == c
        color = "lightgrey" if c == -1 else PALETTE[c % len(PALETTE)]
        ax.scatter(Z[mask, 0], Z[mask, 1], s=55, color=color,
                   edgecolor="white", label=f"cluster {c}")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")
    return Z


def cluster_scatter_2d(df, labels, title, fname):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for c in sorted(set(labels)):
        mask = labels == c
        color = "lightgrey" if c == -1 else PALETTE[c % len(PALETTE)]
        name = "noise" if c == -1 else f"cluster {c} (n={mask.sum()})"
        ax.scatter(df.loc[mask, INCOME_COL], df.loc[mask, SPEND_COL], s=70,
                   color=color, edgecolor="white", alpha=0.85, label=name)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(INCOME_COL)
    ax.set_ylabel(SPEND_COL)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------- pipeline
def main():
    from data_utils import get_clean_data

    df = get_clean_data()
    results = {}

    # ---- feature spaces -------------------------------------------------
    X_plane = df[PLANE_FEATURES].copy()
    X_ext = df[EXTENDED_FEATURES].copy()
    Xp, scaler = scale_features(X_plane)   # saved pipeline operates here
    Xe, _ = scale_features(X_ext)

    print("[1/6] k selection study (income x spending plane) ...")
    k_table = k_selection_study(Xp)
    results["k_selection_plane"] = k_table.to_dict(orient="records")

    print("[2/6] K-Means (plane) ...")
    km, km_labels = fit_kmeans(Xp)
    results["kmeans_plane"] = internal_metrics(Xp, km_labels)
    silhouette_plot(Xp, km_labels, "K-Means (k=5) silhouette - plane",
                    "fig10_silhouette_kmeans.png")
    df["Cluster_KMeans"] = km_labels
    joblib.dump({"model": km, "scaler": scaler, "features": PLANE_FEATURES},
                MODEL_DIR / "kmeans_pipeline.joblib")

    print("[3/6] Hierarchical Ward (plane) ...")
    hc, hc_labels = fit_hierarchical(Xp)
    results["hierarchical_plane"] = internal_metrics(Xp, hc_labels)
    silhouette_plot(Xp, hc_labels, "Agglomerative Ward (k=5) silhouette - plane",
                    "fig15_silhouette_hierarchical.png")

    print("[4/6] Gaussian Mixture (plane) ...")
    gmm, gmm_labels = fit_gmm(Xp)
    results["gmm_plane"] = {**internal_metrics(Xp, gmm_labels),
                            "converged": bool(gmm.converged_)}
    df["Cluster_GMM"] = gmm_labels

    print("[5/6] DBSCAN (plane, tuned on eps x min_samples) ...")
    best_db = tune_dbscan(Xp)
    results["dbscan_tuning"] = best_db
    if best_db:
        _, db_labels = fit_dbscan(Xp, best_db["eps"], best_db["min_samples"])
        results["dbscan_plane"] = internal_metrics(Xp, db_labels)
        df["Cluster_DBSCAN"] = db_labels
        cluster_scatter_2d(df, db_labels,
                           f"DBSCAN (eps={best_db['eps']}, min_samples="
                           f"{best_db['min_samples']}): "
                           f"{best_db['n_clusters']} dense regions, "
                           f"{best_db['noise_frac']:.0%} noise",
                           "fig16_dbscan_plane.png")

    cluster_scatter_2d(df, km_labels,
                       "K-Means segments in the business plane (raw units)",
                       "fig14_kmeans_business_plane.png")

    print("[6/6] Extended space (with Age): metrics + PCA + t-SNE ...")
    km_e, km_e_labels = fit_kmeans(Xe)
    hc_e, hc_e_labels = fit_hierarchical(Xe)
    gmm_e, gmm_e_labels = fit_gmm(Xe)
    results["extended_3d"] = {
        "kmeans": internal_metrics(Xe, km_e_labels),
        "hierarchical": internal_metrics(Xe, hc_e_labels),
        "gmm": internal_metrics(Xe, gmm_e_labels),
    }
    df["Cluster_KMeans_3D"] = km_e_labels
    k_selection_study(Xe, tag="_extended")
    silhouette_plot(Xe, km_e_labels, "K-Means (k=5) silhouette - extended (with Age)",
                    "fig17_silhouette_extended.png")
    pca, explained = pca_analysis(Xe, ["Age", "Income", "Spending"])
    results["pca_explained_variance"] = [float(v) for v in explained]
    tsne_projection(Xe, km_e_labels,
                    "t-SNE map of the extended feature space (Age+Income+Spending)",
                    "fig13_tsne.png")

    # Profiles -> the marketing story of each segment.
    profile = (df.groupby("Cluster_KMeans")
                 .agg(n=("Age", "size"),
                      age_mean=("Age", "mean"),
                      income_mean=("Annual Income (k$)", "mean"),
                      spend_mean=("Spending Score (1-100)", "mean"),
                      male_pct=("Gender", lambda s: 100 * (s == "Male").mean()))
                 .round(1).sort_values("spend_mean", ascending=False))
    results["kmeans_profile"] = json.loads(profile.to_json(orient="records"))

    (REPORT_DIR / "model_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    profile.to_csv(REPORT_DIR / "cluster_profiles.csv")
    df.to_csv(REPORT_DIR / "scored_customers.csv", index=False)

    print("\nModel comparison - plane (Income + Spending):")
    for name in ("kmeans_plane", "hierarchical_plane", "gmm_plane", "dbscan_plane"):
        if name in results:
            print(f"  {name:20s} {results[name]}")
    print("Extended space (Age + Income + Spending):")
    for name in ("kmeans", "hierarchical", "gmm"):
        if name in results.get("extended_3d", {}):
            print(f"  {name:20s} {results['extended_3d'][name]}")
    print("\nSaved: model_comparison.json, cluster_profiles.csv, scored_customers.csv")
    return results


if __name__ == "__main__":
    main()
