# syntax=docker/dockerfile:1.7
# API image: CPU-only PyTorch (keeps the image ~2 GB instead of ~6 GB with CUDA wheels).
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/hf-cache HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# CPU-only torch: resolve torch from the PyTorch CPU index so sentence-transformers never pulls CUDA wheels.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.4" ".[ml,neo4j,anthropic,openai,plots]"
COPY configs ./configs
COPY data/corpus ./data/corpus
COPY data/evaluation ./data/evaluation
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && useradd -m -u 1000 app && mkdir -p /app/data /app/results \
    && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD curl -fsS http://localhost:8000/health || exit 1
ENTRYPOINT ["/entrypoint.sh"]
CMD ["agvrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
