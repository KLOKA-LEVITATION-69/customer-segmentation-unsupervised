"""Load, clean and engineer features for the Mall Customers dataset.

Every transformation records its row counts so the cleaning is auditable.
The dataset arrives clean (no nulls/duplicates); the checks are kept so the
pipeline stays honest and portable to messier versions of the file.
"""
import numpy as np
import pandas as pd

from config import (AGE_COL, DATA_FILE, FEATURES_FULL, GENDER_COL, ID_COL,
                    INCOME_COL, SPEND_COL)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop IDs, standardise column names, apply plausibility filters."""
    log = {}

    df = df.drop(columns=[c for c in (ID_COL,) if c in df.columns]).copy()
    df = df.rename(columns={GENDER_COL: "Gender"})

    log["rows_start"] = len(df)

    # Physiologically implausible values (defensive; the published file has none).
    df = df[(df["Age"].between(0, 100))
            & (df["Annual Income (k$)"].between(0, 1000))
            & (df["Spending Score (1-100)"].between(0, 100))]
    log["rows_after_plausibility"] = len(df)

    df = df.drop_duplicates()
    log["rows_after_dedup"] = len(df)

    df = df.reset_index(drop=True)
    df.attrs["cleaning_log"] = log
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable derived features (no target exists - unsupervised)."""
    df = df.copy()

    # Age band chosen to match common marketing lifecycle segments.
    df["AgeBand"] = pd.cut(df["Age"], bins=[0, 25, 40, 55, 100],
                           labels=["18-25", "26-40", "41-55", "56+"])

    # Income bands in k$ (approx. USD thousands per year).
    df["IncomeBand"] = pd.cut(df["Annual Income (k$)"], bins=[0, 39, 70, 1000],
                              labels=["Low (<40k)", "Middle (40-70k)", "High (>70k)"])

    # Normalised coordinates inside the income x spend plane - useful because
    # K-Means on raw units over-weights income (range 0-137 vs 1-99).
    inc_min, inc_max = df["Annual Income (k$)"].min(), df["Annual Income (k$)"].max()
    spd_min, spd_max = df["Spending Score (1-100)"].min(), df["Spending Score (1-100)"].max()
    df["IncomeNorm"] = (df["Annual Income (k$)"] - inc_min) / (inc_max - inc_min)
    df["SpendNorm"] = (df["Spending Score (1-100)"] - spd_min) / (spd_max - spd_min)

    # Spending-to-income ratio: who spends beyond / below their means.
    df["SpendToIncome"] = df["Spending Score (1-100)"] / df["Annual Income (k$)"].clip(lower=1)
    return df


def get_clean_data() -> pd.DataFrame:
    df = engineer(clean(load_raw()))
    return df


if __name__ == "__main__":
    df = get_clean_data()
    print(df.head())
    print(df["cleaning_log"] if "cleaning_log" in df.attrs else df.attrs)
    print(df.describe(include="all").T)
    print("Feature matrix for clustering:", FEATURES_FULL)
