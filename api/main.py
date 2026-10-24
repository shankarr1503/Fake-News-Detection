import logging
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import joblib
import torch
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from api.config import Settings, load_settings
from api.middleware import BodySizeLimitMiddleware, SecurityHeadersMiddleware
from common.labels import LABEL_REAL
from common.model_meta import read_metadata
from common.paths import AI_MODEL_DIR, AI_MODEL_META, ML_MODEL, ML_MODEL_META, ROOT
from common.text import clean_text, combine_title_text

# uvicorn configures this logger, so messages show up in the server output.
logger = logging.getLogger("uvicorn.error")

# Populated by load_models() at startup; None means that model is unavailable.
ml_model = None
ml_meta = None
ai_tokenizer = None
ai_model = None
ai_meta = None

# One transformer forward pass at a time per worker: concurrent passes would just compete
# for the same CPU cores. Scale throughput with more workers instead.
ai_lock = threading.Lock()


def load_models(torch_threads=0):
    global ml_model, ml_meta, ai_tokenizer, ai_model, ai_meta

    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    # Only load model files you produced yourself: joblib/pickle can execute code on load.
    if not ML_MODEL.exists():
        logger.warning("%s not found; ML predictions unavailable", ML_MODEL)
    else:
        try:
            ml_model = joblib.load(ML_MODEL)
            ml_meta = read_metadata(ML_MODEL_META, "ML", logger)
            logger.info("ML model loaded from %s", ML_MODEL)
        except Exception:
            ml_model = ml_meta = None
            logger.exception("Could not load ML model from %s; ML predictions unavailable", ML_MODEL)

    if not AI_MODEL_DIR.exists():
        logger.warning("%s not found; AI predictions unavailable", AI_MODEL_DIR)
        return
    try:
        ai_tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_DIR)
        ai_model = AutoModelForSequenceClassification.from_pretrained(AI_MODEL_DIR)
        ai_model.eval()
        ai_meta = read_metadata(AI_MODEL_META, "AI", logger)
        logger.info("AI model loaded from %s", AI_MODEL_DIR)
    except Exception:
        ai_tokenizer = ai_model = ai_meta = None
        logger.exception("Could not load AI model from %s; AI predictions unavailable", AI_MODEL_DIR)


class NewsInput(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)]
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100_000)]


class ModelPrediction(BaseModel):
    label: Literal["REAL", "FAKE", "UNAVAILABLE"]
    # Probability of the predicted label; None when the model is unavailable.
    confidence: float | None


class PredictionResponse(BaseModel):
    ml: ModelPrediction
    ai: ModelPrediction


UNAVAILABLE = ModelPrediction(label="UNAVAILABLE", confidence=None)


def to_prediction(prob_real):
    if prob_real > 0.5:
        return ModelPrediction(label="REAL", confidence=prob_real)
    return ModelPrediction(label="FAKE", confidence=1 - prob_real)


router = APIRouter()


def model_status(model, meta):
    return {"loaded": model is not None, "metadata": meta}


@router.get("/health")
def health():
    """Liveness: the process is up. Also reports which models are loaded."""
    return {"status": "ok", "models": {"ml": model_status(ml_model, ml_meta), "ai": model_status(ai_model, ai_meta)}}


@router.get("/ready")
def ready():
    """Readiness: 200 once at least one model can serve predictions, 503 otherwise."""
    is_ready = ml_model is not None or ai_model is not None
    return JSONResponse({"ready": is_ready}, status_code=200 if is_ready else 503)


# A plain `def` so FastAPI runs the CPU-bound inference in a worker thread
# instead of blocking the event loop.
@router.post("/predict", response_model=PredictionResponse)
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
        with ai_lock, torch.no_grad():
            logits = ai_model(**inputs).logits
        prob_real = torch.softmax(logits, dim=1)[0][LABEL_REAL].item()
        ai_result = to_prediction(prob_real)

    return PredictionResponse(ml=ml_result, ai=ai_result)


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        load_models(settings.torch_threads)
        if ml_model is None and ai_model is None:
            logger.error("No models loaded; /ready will report 503 until models are trained and the server restarted")
        yield

    app = FastAPI(
        title="Fake News Detection API",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )

    # Middleware added last runs first, so oversized requests are refused before anything else touches them.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)

    app.include_router(router)
    # Serve the frontend from the same origin, so http://localhost:8000/ works without CORS.
    # Mounted last so it doesn't shadow the API routes.
    app.mount("/", StaticFiles(directory=ROOT / "app", html=True), name="app")
    return app


settings = load_settings()
app = create_app(settings)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
