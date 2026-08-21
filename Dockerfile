# ==============================================================================
# Production Multi-Stage Dockerfile for Plant Disease Classifier
# ==============================================================================

FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    TF_ENABLE_ONEDNN_OPTS=0

# Install system dependencies (libglib2.0, libgl1 for OpenCV/PIL if needed, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy project source code, models, templates, and static assets
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY app.py setup.py plant_disease_efficientnetb0_final.keras ./

# Create logs directory
RUN mkdir -p logs && chmod 777 logs

# Expose server port
EXPOSE 8000

# Docker Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI ASGI server with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
