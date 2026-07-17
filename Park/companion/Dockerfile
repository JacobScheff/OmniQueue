# Multi-stage: build SPA, then run FastAPI + torch on CPU.
# Does not build the C++ DES — companion only needs Python + the PPO checkpoint.
FROM node:22-bookworm AS web
WORKDIR /web
COPY Park/companion/web/package.json Park/companion/web/package-lock.json* ./
RUN npm install
COPY Park/companion/web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app

# Minimal runtime deps (no pygame / native DES)
RUN pip install --no-cache-dir \
      "numpy>=1.24" \
      "networkx>=3.0" \
      "torch>=2.0" \
      "fastapi>=0.110" \
      "uvicorn[standard]>=0.27" \
      "httpx>=0.27" \
      "pydantic>=2.0" \
    && rm -rf /root/.cache/pip

COPY Park/ /app/Park/
COPY --from=web /web/dist /app/Park/companion/web/dist

# Default model path matches companion/settings.py (override by editing settings
# or mounting a file over /app/Park/companion/model/ppo_live.pt).
EXPOSE 8000
CMD ["python", "/app/Park/run_companion.py"]
