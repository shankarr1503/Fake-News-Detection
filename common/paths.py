"""Filesystem locations, resolved from the repo root so scripts work from any CWD."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "news.csv"
PROCESSED_DATA = DATA_DIR / "processed_news.csv"

MODELS_DIR = ROOT / "models"
ML_MODEL = MODELS_DIR / "ml_pipeline.joblib"
AI_MODEL_DIR = MODELS_DIR / "ai_model"

RESULTS_DIR = ROOT / "results"
