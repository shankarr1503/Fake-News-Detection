import pytest
import torch
from fastapi.testclient import TestClient

from api import main

PAYLOAD = {"title": "Some headline", "text": "Some article body."}


class StubMLModel:
    classes_ = [0, 1]

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


@pytest.fixture
def client(monkeypatch):
    # No `with` block, so the lifespan (which loads real models from disk) doesn't run.
    monkeypatch.setattr(main, "ml_model", None)
    monkeypatch.setattr(main, "ai_model", None)
    monkeypatch.setattr(main, "ai_tokenizer", None)
    return TestClient(main.app)


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


@pytest.mark.parametrize("payload", [
    {"title": "", "text": "body"},
    {"title": "   ", "text": "body"},
    {"title": "title"},
    {"title": "title", "text": "x" * 100_001},
])
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_cors_preflight_is_allowed(client):
    response = client.options("/predict", headers={
        "Origin": "http://example.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] in ("*", "http://example.com")


def test_frontend_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>" in response.text


def test_health(client):
    assert client.get("/health").json() == {"ml_model_loaded": False, "ai_model_loaded": False}
