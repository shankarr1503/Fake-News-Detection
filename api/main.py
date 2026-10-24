import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Optional

import joblib
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.data import LABEL_REAL
from common.paths import AI_MODEL_DIR, ML_MODEL, ROOT
from common.text import clean_text, combine_title_text

# uvicorn configures this logger, so messages show up in the server output.
logger = logging.getLogger("uvicorn.error")

# Populated by load_models() at startup; None means that model is unavailable.
ml_model = None
ai_tokenizer = None
ai_model = None


def load_models():
    global ml_model, ai_tokenizer, ai_model

    # Only load model files you produced yourself: joblib/pickle can execute code on load.
    try:
        ml_model = joblib.load(ML_MODEL)
        logger.info("ML model loaded from %s", ML_MODEL)
    except Exception:
        ml_model = None
        logger.exception("Could not load ML model from %s; ML predictions unavailable", ML_MODEL)

    if not AI_MODEL_DIR.exists():
        logger.warning("%s not found; AI predictions unavailable", AI_MODEL_DIR)
        return
    try:
        ai_tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_DIR)
        ai_model = AutoModelForSequenceClassification.from_pretrained(AI_MODEL_DIR)
        ai_model.eval()
        logger.info("AI model loaded from %s", AI_MODEL_DIR)
    except Exception:
        ai_tokenizer = ai_model = None
        logger.exception("Could not load AI model from %s; AI predictions unavailable", AI_MODEL_DIR)


@asynccontextmanager
async def lifespan(app):
    load_models()
    yield


app = FastAPI(title="Fake News Detection API", lifespan=lifespan)

# Comma-separated list of allowed origins, e.g. "http://localhost:5500". "*" allows any.
cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class NewsInput(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)]
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100_000)]


class ModelPrediction(BaseModel):
    label: Literal["REAL", "FAKE", "UNAVAILABLE"]
    # Probability of the predicted label; None when the model is unavailable.
    confidence: Optional[float]


class PredictionResponse(BaseModel):
    ml: ModelPrediction
    ai: ModelPrediction


UNAVAILABLE = ModelPrediction(label="UNAVAILABLE", confidence=None)


def to_prediction(prob_real):
    if prob_real > 0.5:
        return ModelPrediction(label="REAL", confidence=prob_real)
    return ModelPrediction(label="FAKE", confidence=1 - prob_real)


@app.get("/health")
def health():
    return {"ml_model_loaded": ml_model is not None, "ai_model_loaded": ai_model is not None}


# A plain `def` so FastAPI runs the CPU-bound inference in a worker thread
# instead of blocking the event loop.
@app.post("/predict", response_model=PredictionResponse)
def predict(news: NewsInput):
    combined_text = combine_title_text(news.title, news.text)

    ml_result = UNAVAILABLE
    if ml_model is not None:
        classes = list(ml_model.classes_)
        prob_real = float(ml_model.predict_proba([clean_text(combined_text)])[0][classes.index(LABEL_REAL)])
        ml_result = to_prediction(prob_real)

    ai_result = UNAVAILABLE
    if ai_model is not None:
        inputs = ai_tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = ai_model(**inputs).logits
        prob_real = torch.softmax(logits, dim=1)[0][LABEL_REAL].item()
        ai_result = to_prediction(prob_real)

    return PredictionResponse(ml=ml_result, ai=ai_result)


# Serve the frontend from the same origin, so http://localhost:8000/ works without CORS.
# Mounted last so it doesn't shadow the API routes.
app.mount("/", StaticFiles(directory=ROOT / "app", html=True), name="app")


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
