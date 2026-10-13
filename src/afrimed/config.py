import os
from pathlib import Path

ROOT = Path(os.environ.get("AFRIMED_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "configs"

DATASET_ID = "intronhealth/afrimedqa_v2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
LETTERS = ["A", "B", "C", "D", "E"]


def ensure_dirs():
    for path in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
