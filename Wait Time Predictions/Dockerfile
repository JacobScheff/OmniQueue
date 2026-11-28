FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MALLOC_ARENA_MAX=2
COPY requirements*.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch>=2.7,<3' \
    && pip install --no-cache-dir -r requirements-forecast.txt
COPY api api
COPY ml ml
COPY view.py ./
RUN useradd --system app
USER app
CMD ["sh", "-c", "exec gunicorn --worker-class gthread --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT ml.forecast_app:app"]
