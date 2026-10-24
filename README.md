# Fake News Detection

Classifies a news article (title + body) as REAL or FAKE with two models side by side:

- **ML**: TF-IDF + Logistic Regression (scikit-learn)
- **AI**: fine-tuned DistilBERT (Hugging Face transformers)

A FastAPI backend serves both models and a small web frontend.

> **Limitations:** the models are trained on [`fake_or_real_news.csv`](https://github.com/lutzhamel/fake-news),
> about 6,300 political articles from around the 2016 US election. They learn the writing style and topics of
> that dataset, not whether a claim is true, and they generalise poorly to other topics or periods.
> Treat the output as a hint, not a fact-check.

## Setup

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # or requirements.txt without the test tools
```

## Usage

Run every command from the repository root, using `python -m` so the shared `common` package can be imported.

```bash
python -m scripts.download_data   # -> data/news.csv
python -m scripts.preprocess      # -> data/processed_news.csv
python -m models.train_ml         # -> models/ml_pipeline.joblib (about a minute)
python -m models.train_ai         # -> models/ai_model/ (slow on CPU; see --help for options)
python -m api.main                # serves the API and the frontend on http://127.0.0.1:8000
```

Then open <http://127.0.0.1:8000/> in a browser. The API starts even if only one model is trained; the other one
is reported as `UNAVAILABLE`.

`train_ai` options: `--train-samples N` (default 2000; `0` uses the full training split), `--epochs N`
(default 1) and `--base-model NAME_OR_PATH`. Both models are evaluated on the same stratified test split
(`random_state=42`), so their scores can be compared.

### Configuration (environment variables)

| Variable       | Default     | Purpose                                                  |
| -------------- | ----------- | -------------------------------------------------------- |
| `HOST`         | `127.0.0.1` | Interface to bind. Use `0.0.0.0` to expose it on the network. |
| `PORT`         | `8000`      | Port to listen on.                                       |
| `CORS_ORIGINS` | `*`         | Comma-separated origins allowed to call the API.         |

## API

`POST /predict`

```json
{ "title": "Headline", "text": "Article body" }
```

```json
{
  "ml": { "label": "FAKE", "confidence": 0.91 },
  "ai": { "label": "UNAVAILABLE", "confidence": null }
}
```

`label` is `REAL`, `FAKE` or `UNAVAILABLE` (model not loaded). `confidence` is the probability of the predicted
label. `title` (1–1,000 chars) and `text` (1–100,000 chars) are required; invalid input returns `422`.

`GET /health` reports which models are loaded.

## Tests

```bash
python -m pytest                     # unit tests; no trained models needed
python -m scripts.smoke_test_api     # against a running server
```

## Project layout

```
api/main.py          FastAPI app (also serves app/)
app/index.html       Frontend
common/              Paths, text cleaning and data split shared by training and serving
models/train_*.py    Training scripts (trained models are written next to them, git-ignored)
scripts/             Download, preprocessing and API smoke test
tests/               pytest suite
```

Model files are loaded with `joblib`/`pickle`, which can run code. Only load models you trained yourself.
Retrain after upgrading scikit-learn or transformers.
