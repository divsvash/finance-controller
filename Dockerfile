# finance-controller -- reproducible production image
# Build context must contain pyproject.toml + package source only
# (.dockerignore keeps secrets/test artifacts out of layers).

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the project from pyproject.toml (reproducible; layer-cached).
COPY pyproject.toml README.md ./
COPY finance_controller ./finance_controller
RUN pip install --no-cache-dir .

# Non-root user; /app/data/runs is the only writable app location.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/runs \
    && chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen(\
        'http://127.0.0.1:8000/health', timeout=2).status==200 else 1)" \
        || exit 1

# Production command. Deterministic mode needs no env vars;
# FINANCE_LLM_* are supplied at `docker run -e` time only.
CMD ["uvicorn", "finance_controller.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
