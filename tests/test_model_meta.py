import json
import logging

from common import model_meta


def test_metadata_round_trip(tmp_path):
    path = tmp_path / "model.meta.json"
    model_meta.write_metadata(path, metrics={"accuracy": 0.9}, train_size=10)

    meta = model_meta.read_metadata(path, "ML")

    assert meta["metrics"] == {"accuracy": 0.9}
    assert meta["train_size"] == 10
    assert set(meta["library_versions"]) == set(model_meta.TRACKED_PACKAGES)


def test_warns_when_library_version_changed(tmp_path, monkeypatch, caplog):
    path = tmp_path / "model.meta.json"
    path.write_text(json.dumps({"library_versions": {"scikit-learn": "1.0.0", "torch": "2.14.1+cu130"}}))
    monkeypatch.setattr(model_meta, "_installed_versions", lambda: {"scikit-learn": "1.9.1", "torch": "2.14.1+cpu"})

    with caplog.at_level(logging.WARNING):
        model_meta.read_metadata(path, "ML")

    # scikit-learn changed; torch differs only in its build label, which doesn't matter.
    assert "scikit-learn 1.0.0" in caplog.text
    assert "torch" not in caplog.text


def test_missing_metadata_returns_none(tmp_path):
    assert model_meta.read_metadata(tmp_path / "missing.json", "ML") is None
