"""Central configuration: paths, feature groups and random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
MODEL_DIR = OUT_DIR / "models"

for _d in (DATA_DIR, FIG_DIR, REPORT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "Mall_Customers.csv"
RANDOM_STATE = 42

ID_COL = "CustomerID"
GENDER_COL = "Genre"          # gender column is named "Genre" in this dataset
AGE_COL = "Age"
INCOME_COL = "Annual Income (k$)"
SPEND_COL = "Spending Score (1-100)"

FEATURES_2D = [INCOME_COL, SPEND_COL]          # classic 2-feature analysis
FEATURES_FULL = [AGE_COL, INCOME_COL, SPEND_COL]  # extended 3-feature analysis
