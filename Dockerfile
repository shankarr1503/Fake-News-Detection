# Serving image for the Fake News Detection API (CPU-only).
# Trained models are not baked in: mount them at /models (see docker-compose.yml).
ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch wheels are a fraction of the size of the default CUDA build.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --index-url "${TORCH_INDEX_URL}" "torch==$(sed -n 's/^torch==//p' requirements.txt)" \
    && pip install -r requirements.txt

RUN useradd --system --uid 10001 --no-create-home appuser

COPY common/ common/
COPY api/ api/
COPY app/ app/

ENV MODELS_DIR=/models \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    UVICORN_WORKERS=1 \
    HF_HUB_OFFLINE=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"UVICORN_PORT\"]}/ready', timeout=4)"]

# uvicorn reads UVICORN_* environment variables for its options.
CMD ["uvicorn", "api.main:app", "--proxy-headers", "--no-server-header"]
