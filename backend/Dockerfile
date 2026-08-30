# Multi-stage Dockerfile for RXNGraphormer Backend
# Build stage: install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install to user site
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
    -f https://data.pyg.org/whl/torch-2.1.2+cpu.html

# Runtime stage: minimal image
FROM python:3.11-slim

WORKDIR /app

# Install runtime system dependencies (RDKit needs some)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY backend/ ./backend/
COPY rxngraphormer/ ./rxngraphormer/
COPY dataset/ ./dataset/

# Models directory will be populated at runtime by download_models.py
# (checkpoints are large and not baked into the image)
RUN mkdir -p /app/models

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Default command runs the FastAPI server
# download_models.py is called from lifespan in main.py
CMD ["python", "-m", "backend.main"]