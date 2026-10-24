"""Filesystem locations, resolved from the repo root so scripts work from any CWD."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "news.csv"
PROCESSED_DATA = DATA_DIR / "processed_news.csv"

# Overridable so a deployment can mount trained models from elsewhere (e.g. a Docker volume).
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
ML_MODEL = MODELS_DIR / "ml_pipeline.joblib"
ML_MODEL_META = MODELS_DIR / "ml_pipeline.meta.json"
AI_MODEL_DIR = MODELS_DIR / "ai_model"
AI_MODEL_META = AI_MODEL_DIR / "training_meta.json"

RESULTS_DIR = ROOT / "results"
