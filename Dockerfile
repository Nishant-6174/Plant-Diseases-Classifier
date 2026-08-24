# ==============================================================================
# Production Dockerfile for Plant Disease Classifier
# ==============================================================================

FROM python:3.10-slim

# ------------------------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------------------------

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_ENABLE_ONEDNN_OPTS=0

# ------------------------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Working directory
# ------------------------------------------------------------------------------

WORKDIR /app

# ------------------------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------------------------

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ------------------------------------------------------------------------------
# Application files
# ------------------------------------------------------------------------------

COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/

COPY app.py setup.py plant_disease_efficientnetb0_compatible.keras ./

# ------------------------------------------------------------------------------
# Logs
# ------------------------------------------------------------------------------

RUN mkdir -p logs && chmod 777 logs

# ------------------------------------------------------------------------------
# Render listens on its assigned PORT.
# The application itself reads PORT from the environment.
# ------------------------------------------------------------------------------

EXPOSE 10000

# ------------------------------------------------------------------------------
# Healthcheck
# ------------------------------------------------------------------------------

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# ------------------------------------------------------------------------------
# Start FastAPI
# ------------------------------------------------------------------------------

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]