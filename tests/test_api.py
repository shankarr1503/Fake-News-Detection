import json

import joblib
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from api import main
from api.config import Settings
from common.model_meta import write_metadata

PAYLOAD = {"title": "Some headline", "text": "Some article body."}


def make_settings(**overrides):
    defaults = dict(
        cors_origins=[], max_request_bytes=1_000_000, torch_threads=0, docs_enabled=True, host="127.0.0.1", port=8000
    )
    return Settings(**{**defaults, **overrides})


class StubMLModel:
    classes_ = (0, 1)

    def __init__(self, prob_real):
        self.prob_real = prob_real

    def predict_proba(self, texts):
        return [[1 - self.prob_real, self.prob_real] for _ in texts]


class StubTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}


class StubAIModel:
    def __init__(self, logits):
        self.logits = torch.tensor([logits])

    def __call__(self, **inputs):
        return type("Output", (), {"logits": self.logits})()


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    for name in ("ml_model", "ml_meta", "ai_model", "ai_tokenizer", "ai_meta"):
        monkeypatch.setattr(main, name, None)


@pytest.fixture
def client():
    # No `with` block, so the lifespan (which loads real models from disk) doesn't run.
    return TestClient(main.create_app(make_settings()))


def test_predict_without_models_reports_unavailable(client):
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    unavailable = {"label": "UNAVAILABLE", "confidence": None}
    assert response.json() == {"ml": unavailable, "ai": unavailable}


def test_predict_with_models(client, monkeypatch):
    monkeypatch.setattr(main, "ml_model", StubMLModel(prob_real=0.2))
    monkeypatch.setattr(main, "ai_tokenizer", StubTokenizer())
    monkeypatch.setattr(main, "ai_model", StubAIModel([0.0, 2.0]))

    body = client.post("/predict", json=PAYLOAD).json()

    assert body["ml"] == {"label": "FAKE", "confidence": pytest.approx(0.8)}
    assert body["ai"]["label"] == "REAL"
    assert body["ai"]["confidence"] == pytest.approx(torch.softmax(torch.tensor([0.0, 2.0]), 0)[1].item())


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "", "text": "body"},
        {"title": "   ", "text": "body"},
        {"title": "title"},
        {"title": "title", "text": "x" * 100_001},
    ],
)
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_oversized_body_is_rejected_before_parsing():
    client = TestClient(main.create_app(make_settings(max_request_bytes=100)))
    response = client.post("/predict", json={"title": "t", "text": "x" * 200})
    assert response.status_code == 413


def test_body_without_content_length_is_rejected(client):
    def chunks():
        yield json.dumps(PAYLOAD).encode()

    response = client.post("/predict", content=chunks(), headers={"Content-Type": "application/json"})
    assert response.status_code == 411


PREFLIGHT_HEADERS = {
    "Origin": "http://example.com",
    "Access-Control-Request-Method": "POST",
    "Access-Control-Request-Headers": "content-type",
}


def test_cross_origin_requests_are_not_allowed_by_default(client):
    response = client.options("/predict", headers=PREFLIGHT_HEADERS)
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_configured_origins():
    client = TestClient(main.create_app(make_settings(cors_origins=["http://example.com"])))
    response = client.options("/predict", headers=PREFLIGHT_HEADERS)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://example.com"


def test_frontend_is_served_with_security_headers(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>" in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    csp = response.headers["content-security-policy"]
    assert "script-src 'self';" in csp
    assert "unsafe-inline" not in csp


def test_docs_are_exempt_from_csp_and_can_be_disabled(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "content-security-policy" not in response.headers

    no_docs = TestClient(main.create_app(make_settings(docs_enabled=False)))
    assert no_docs.get("/docs").status_code == 404
    assert no_docs.get("/openapi.json").status_code == 404


def test_health_reports_model_status(client, monkeypatch):
    monkeypatch.setattr(main, "ml_model", StubMLModel(0.5))
    monkeypatch.setattr(main, "ml_meta", {"metrics": {"accuracy": 0.9}})

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["models"]["ml"] == {"loaded": True, "metadata": {"metrics": {"accuracy": 0.9}}}
    assert body["models"]["ai"] == {"loaded": False, "metadata": None}


def test_ready_requires_a_loaded_model(client, monkeypatch):
    assert client.get("/ready").status_code == 503
    monkeypatch.setattr(main, "ml_model", StubMLModel(0.5))
    assert client.get("/ready").status_code == 200


def test_load_models_handles_missing_and_present_files(tmp_path, monkeypatch):
    ml_path = tmp_path / "ml_pipeline.joblib"
    monkeypatch.setattr(main, "ML_MODEL", ml_path)
    monkeypatch.setattr(main, "ML_MODEL_META", tmp_path / "ml_pipeline.meta.json")
    monkeypatch.setattr(main, "AI_MODEL_DIR", tmp_path / "ai_model")

    main.load_models()
    assert main.ml_model is None
    assert main.ai_model is None

    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipeline.fit(["senate passes budget", "aliens built the pyramids"], [1, 0])
    joblib.dump(pipeline, ml_path)
    write_metadata(tmp_path / "ml_pipeline.meta.json", metrics={"accuracy": 1.0})

    main.load_models()
    assert main.ml_model is not None
    assert main.ml_meta["metrics"] == {"accuracy": 1.0}
    assert TestClient(main.create_app(make_settings())).post("/predict", json=PAYLOAD).json()["ml"]["label"] in (
        "REAL",
        "FAKE",
    )
