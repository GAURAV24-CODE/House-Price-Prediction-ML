from pathlib import Path

# ============================
# Project Directory Structure
# ============================

# Root project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = BASE_DIR / "data"

# Models directory
MODELS_DIR = BASE_DIR / "models" / "trained_models"

# Reports directory
REPORTS_DIR = BASE_DIR / "reports"

# Dataset path
DATASET_PATH = DATA_DIR / "kc_house_data.csv"

FIGURES_DIR = REPORTS_DIR / "figures"

CSV_DIR = REPORTS_DIR / "csv"