# Multi-stage: build SPA, then run FastAPI + ONNX Runtime on CPU.
# Does not build the C++ DES — companion only needs Python + ONNX checkpoints.
# Build from repo root:
#   docker build -f Park/companion/Dockerfile -t omniqueue-companion .
FROM node:22-bookworm AS web
WORKDIR /web
COPY Park/companion/web/package.json Park/companion/web/package-lock.json* ./
RUN npm ci
COPY Park/companion/web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app

# onnxruntime (not torch) keeps the free-tier image small enough for ~512MB hosts.
RUN pip install --no-cache-dir \
      "numpy>=1.24" \
      "networkx>=3.0" \
      "onnxruntime>=1.17" \
      "fastapi>=0.110" \
      "uvicorn[standard]>=0.27" \
      "httpx>=0.27" \
      "pydantic>=2.0" \
    && rm -rf /root/.cache/pip

COPY Park/ /app/Park/
COPY --from=web /web/dist /app/Park/companion/web/dist

# Drop training-only .pt weights from the image (ONNX is what runtime loads).
RUN rm -f /app/Park/companion/model/*.pt \
    && mkdir -p /app/Park/cache \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV HOST=0.0.0.0 \
    PORT=8000 \
    COMPANION_DEVICE=cpu \
    HOME=/home/appuser \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python", "/app/Park/run_companion.py"]
