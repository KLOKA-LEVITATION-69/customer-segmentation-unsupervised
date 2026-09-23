"""Exploratory data analysis: 8 figures + a text summary of key insights.

Run:  python src/eda.py
Outputs land in outputs/figures and outputs/reports.
"""
import io
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (AGE_COL, FIG_DIR, INCOME_COL, REPORT_DIR, SPEND_COL)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
ACCENT = ["#1f77b4", "#d62728"]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def fig01_overview(df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Mall Customers - Dataset Overview (n=200)", fontsize=15, fontweight="bold")

    counts = df["Gender"].value_counts()
    axes[0, 0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                   colors=ACCENT, startangle=90, wedgeprops=dict(edgecolor="white"))
    axes[0, 0].set_title("Gender split")

    sns.histplot(df, x=AGE_COL, bins=18, kde=True, ax=axes[0, 1], color="#1f77b4")
    axes[0, 1].set_title("Age distribution")

    sns.histplot(df, x=INCOME_COL, bins=18, kde=True, ax=axes[0, 2], color="#2ca02c")
    axes[0, 2].set_title("Annual income (k$)")

    sns.histplot(df, x=SPEND_COL, bins=18, kde=True, ax=axes[1, 0], color="#ff7f0e")
    axes[1, 0].set_title("Spending score (1-100)")

    sns.scatterplot(df, x=INCOME_COL, y=SPEND_COL, ax=axes[1, 1],
                    hue="Gender", palette=ACCENT, alpha=0.8)
    axes[1, 1].set_title("Income vs spending - the clustering plane")

    sns.scatterplot(df, x=AGE_COL, y=SPEND_COL, ax=axes[1, 2],
                    hue="Gender", palette=ACCENT, alpha=0.8)
    axes[1, 2].set_title("Age vs spending")

    _save(fig, "fig01_overview.png")


def fig02_gender_boxplots(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title in zip(
            axes,
            [AGE_COL, INCOME_COL, SPEND_COL],
            ["Age by gender", "Income by gender (k$)", "Spending score by gender"]):
        sns.boxplot(df, x="Gender", y=col, hue="Gender", ax=ax,
                    palette=ACCENT, legend=False)
        ax.set_title(title)
    _save(fig, "fig02_gender_boxplots.png")


def fig03_04_heatmaps(df):
    num = df[[AGE_COL, INCOME_COL, SPEND_COL, "SpendToIncome"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.heatmap(num.corr(method="pearson"), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, ax=axes[0])
    axes[0].set_title("Pearson correlation (linear)")
    sns.heatmap(num.corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, ax=axes[1])
    axes[1].set_title("Spearman correlation (monotonic)")
    _save(fig, "fig03_04_correlation_heatmaps.png")


def fig05_pairplot(df):
    g = sns.pairplot(df[[AGE_COL, INCOME_COL, SPEND_COL, "Gender"]],
                     hue="Gender", palette=ACCENT, diag_kind="kde", height=2.4)
    g.figure.suptitle("Pairwise relationships", y=1.02, fontweight="bold")
    g.figure.savefig(FIG_DIR / "fig05_pairplot.png", bbox_inches="tight")
    plt.close(g.figure)
    print("  saved fig05_pairplot.png")


def fig06_income_spend_gender(df):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.scatterplot(df, x=INCOME_COL, y=SPEND_COL, hue="Gender",
                    palette=ACCENT, s=70, alpha=0.85, edgecolor="white", ax=ax)
    ax.set_title("The classic 5-blob structure: income vs spending score",
                 fontweight="bold")
    # Visual guide to the five human-interpretable zones.
    ax.axvline(70, color="grey", ls="--", lw=1)
    ax.axhline(50, color="grey", ls="--", lw=1)
    ax.text(20, 95, "low income\nhigh spending", ha="center", fontsize=9, color="dimgray")
    ax.text(115, 95, "high income\nhigh spending", ha="center", fontsize=9, color="dimgray")
    ax.text(20, 5, "low income\nlow spending", ha="center", fontsize=9, color="dimgray")
    ax.text(115, 5, "high income\nlow spending", ha="center", fontsize=9, color="dimgray")
    _save(fig, "fig06_income_vs_spend_zones.png")


def fig07_08_bands(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(df, x="AgeBand", hue="Gender", palette=ACCENT, ax=axes[0])
    axes[0].set_title("Customers per age band")
    sns.barplot(df, x="IncomeBand", y=SPEND_COL, hue="Gender",
                palette=ACCENT, ax=axes[1], errorbar=("ci", 95))
    axes[1].set_title("Mean spending score by income band")
    axes[1].tick_params(axis="x", rotation=12)
    _save(fig, "fig07_08_band_bars.png")


def build_report(df):
    lines = []
    lines.append("EDA SUMMARY - Mall Customers segmentation dataset")
    lines.append("=" * 60)
    lines.append(f"Rows: {len(df)} | Columns: {list(df.columns)}")
    lines.append(f"Gender split: {df['Gender'].value_counts().to_dict()}")
    lines.append("")
    lines.append("Central tendencies")
    lines.append(f"  Age            mean={df[AGE_COL].mean():.1f}  median={df[AGE_COL].median():.0f}  "
                 f"range={df[AGE_COL].min()}-{df[AGE_COL].max()}")
    lines.append(f"  Income (k$)    mean={df[INCOME_COL].mean():.1f}  median={df[INCOME_COL].median():.0f}  "
                 f"range={df[INCOME_COL].min()}-{df[INCOME_COL].max()}")
    lines.append(f"  Spending       mean={df[SPEND_COL].mean():.1f}  median={df[SPEND_COL].median():.0f}  "
                 f"range={df[SPEND_COL].min()}-{df[SPEND_COL].max()}")
    lines.append("")
    r = df[[AGE_COL, INCOME_COL, SPEND_COL]].corr(method="pearson")
    lines.append("Pearson correlations (Age / Income / Spending)")
    lines.append(r.round(3).to_string())
    lines.append("")
    young = df[df[AGE_COL] <= 40][SPEND_COL].mean()
    older = df[df[AGE_COL] > 40][SPEND_COL].mean()
    lines.append(f"Mean spending score, age <=40: {young:.1f} | age >40: {older:.1f}")
    hi = df[df[INCOME_COL] > 70]
    lines.append(f"High earners (>70k$): n={len(hi)}, mean spending={hi[SPEND_COL].mean():.1f} "
                 f"(bimodal: some spend a lot, some very little)")
    lines.append("")
    lines.append("Key insight: income vs spending shows ~5 visually separable groups ->")
    lines.append("ideal case for K-Means; age adds nuance (young customers cluster tighter).")
    (REPORT_DIR / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("  saved eda_summary.txt")


def main():
    from data_utils import get_clean_data
    df = get_clean_data()
    print("Running EDA ...")
    fig01_overview(df)
    fig02_gender_boxplots(df)
    fig03_04_heatmaps(df)
    fig05_pairplot(df)
    fig06_income_spend_gender(df)
    fig07_08_bands(df)
    build_report(df)
    print("EDA complete.")


if __name__ == "__main__":
    main()
