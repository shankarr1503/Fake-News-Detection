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

Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

| File                     | Contents                                     |
| ------------------------ | -------------------------------------------- |
| `requirements.txt`       | Serving only (what the Docker image installs) |
| `requirements-train.txt` | + data preparation and training              |
| `requirements-dev.txt`   | + tests, linting and the API smoke test      |

Without a GPU, install the much smaller CPU build of PyTorch first:
`pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1`.

## Training

Run every command from the repository root, using `python -m` so the shared `common` package can be imported.

```bash
python -m scripts.download_data   # -> data/news.csv
python -m scripts.preprocess      # -> data/processed_news.csv
python -m models.train_ml         # -> models/ml_pipeline.joblib (about a minute)
python -m models.train_ai         # -> models/ai_model/ (slow on CPU; see --help)
```

Both models are evaluated on the same stratified test split (`random_state=42`), so their scores are comparable.
A training run **does not save the model** if test accuracy is below `--min-accuracy` (0.85 for ML, 0.80 for
AI), so a bad run can't replace a working model. Each model is saved with a metadata file recording when it was
trained, its test scores and the library versions used; the API logs a warning if those versions differ from
the installed ones.

`train_ai` options: `--train-samples N` (default 2000; `0` uses the full training split), `--epochs N`
(default 1), `--base-model NAME_OR_PATH`, `--min-accuracy X`.

## Running locally

```bash
python -m api.main                # http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000/> for the web page, or <http://127.0.0.1:8000/docs> for the interactive API docs.

## Deploying with Docker

The image contains the API and frontend only. Trained models are mounted from the host at `/models`.

```bash
# Train the models first (see above), then:
docker compose up -d --build
curl http://localhost:8000/ready
```

The image installs CPU-only PyTorch, runs as a non-root user, has a container `HEALTHCHECK` on `/ready`, and
`docker-compose.yml` runs it with a read-only filesystem. To use a registry mirror, pass
`--build-arg BASE_IMAGE=...`.

**In front of the container**, put a reverse proxy or load balancer that terminates TLS and applies rate
limiting. The app doesn't do either itself. uvicorn is started with `--proxy-headers`, so it trusts
`X-Forwarded-*` headers from the proxy.

**Sizing:** each worker process loads its own copy of both models (roughly 0.5–1 GB of RAM with DistilBERT and PyTorch) and runs
one DistilBERT inference at a time. Scale with `UVICORN_WORKERS`, and set `TORCH_THREADS` to about
`CPU cores / workers` so workers don't compete for the same cores.

## Configuration

All settings are environment variables.

| Variable            | Default            | Purpose                                                                  |
| ------------------- | ------------------ | ------------------------------------------------------------------------ |
| `MODELS_DIR`        | `./models`         | Where trained models are loaded from (`/models` in Docker).              |
| `CORS_ORIGINS`      | *(empty)*          | Comma-separated origins allowed to call the API from another site. Empty = same-origin only, which is all the bundled frontend needs. Use `null` to allow the page opened from disk (`file://`). |
| `MAX_REQUEST_BYTES` | `1000000`          | Larger request bodies are rejected with `413`.                           |
| `TORCH_THREADS`     | `0` (all cores)    | PyTorch threads per worker.                                              |
| `DOCS_ENABLED`      | `true`             | Serve `/docs`, `/redoc` and `/openapi.json`.                             |
| `UVICORN_WORKERS`   | `1`                | Worker processes (Docker).                                               |
| `UVICORN_HOST` / `UVICORN_PORT` | `0.0.0.0` / `8000` | Bind address (Docker).                                       |
| `HOST` / `PORT`     | `127.0.0.1` / `8000` | Bind address for `python -m api.main`.                                 |

## API

### `POST /predict`

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
label. `title` (1–1,000 chars) and `text` (1–100,000 chars) are required.

| Status | Meaning                                       |
| ------ | --------------------------------------------- |
| `422`  | Invalid input (details in the response body)  |
| `411`  | Request without a `Content-Length` header     |
| `413`  | Request body larger than `MAX_REQUEST_BYTES`  |

### `GET /health`

Liveness check. Always `200` while the process is running, and reports which models are loaded along with their
training metadata.

### `GET /ready`

Readiness check. `200` once at least one model is loaded, `503` otherwise. Use it for load-balancer and
orchestrator readiness probes.

## Development

```bash
python -m pytest                     # backend unit tests; no trained models needed
ruff check . && ruff format --check .
python -m scripts.smoke_test_api     # against a running server
```

### Frontend tests

Browser tests in `tests/e2e/` use [Playwright](https://playwright.dev/) and Node.js 18+. They start the API
themselves (on port 8765, with no models) and mock `/predict` where they need specific results or errors. They
cover keyboard submission, validation, every error state, timeouts, the result states, the 3D scene's states,
phone-width layouts and an
[axe](https://github.com/dequelabs/axe-core) accessibility audit, and fail on any console error, including
Content-Security-Policy violations.

```bash
npm ci
npx playwright install chromium      # first time only
npx playwright test
```

CI (`.github/workflows/ci.yml`) runs lint, the backend and browser tests, and builds and starts the Docker image, on
every pull request and push to `master`. Dependabot proposes monthly dependency updates. Retrain the models after upgrading
scikit-learn or transformers.

## Project layout

```
api/                 FastAPI app: main.py (routes), config.py (settings), middleware.py
app/                 Frontend (index.html, style.css, app.js, favicon.svg); plain HTML/CSS/JS, no build step
common/              Paths, labels, text cleaning, data split and model metadata, shared by training and serving
models/train_*.py    Training scripts (trained models are written next to them, git-ignored)
scripts/             Download, preprocessing and API smoke test
tests/               pytest suite; tests/e2e/ holds the Playwright browser tests
```

## Security notes

- Model files are loaded with `joblib`/`pickle`, which can run code. Only load models you trained yourself.
- The frontend is served with a strict Content-Security-Policy (no inline scripts or styles) and `X-Frame-Options`, `X-Content-Type-Options`
  and `Referrer-Policy` headers. `/docs` and `/redoc` are exempt from the CSP because they load scripts from a CDN.
  Set `DOCS_ENABLED=false` if you don't want them public.
