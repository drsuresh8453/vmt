# Dockerfile — Vehicle Mileage MLOps Project
# Author: Suresh D R | AI Product Developer & Technology Mentor
# DV Analytics

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY monitoring/ ./monitoring/
COPY tests/ ./tests/

# Create directories
RUN mkdir -p models data/reference data/current reports

# Environment variables (overridden by k8s secrets)
ENV AWS_REGION=ap-south-1
ENV S3_BUCKET=vehicle-mileage-project
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose ports
# 8501 → Streamlit app
# 8000 → FastAPI
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: run Streamlit app
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
