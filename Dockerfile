# Dockerfile for NLP Transformers Examples
# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Set Python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Download spaCy model (for NER)
RUN python -m spacy download en_core_web_sm || true

# Create non-root user
RUN useradd -m -u 1000 nlpuser && chown -R nlpuser:nlpuser /app
USER nlpuser

# Expose ports for all UIs
EXPOSE 7860 7861 7862 7863 7864 7865 7866 7867 7868 7869

# Default command: Show UI launcher menu
CMD ["python", "launch_ui.py"]
