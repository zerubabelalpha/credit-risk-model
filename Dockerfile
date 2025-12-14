# ============================================================
# Dockerfile - Containerize the Credit Risk API
# ============================================================
# This file defines how to build a Docker image for the
# Credit Risk FastAPI application.

# ------------------------------------------------------------
# STEP 1: Base image
# ------------------------------------------------------------
# python:3.11-slim is lightweight and suitable for production
FROM python:3.11-slim

# ------------------------------------------------------------
# STEP 2: Environment settings
# ------------------------------------------------------------
# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure stdout/stderr are flushed immediately
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# STEP 3: Set working directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# STEP 4: Install system dependencies (minimal)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# STEP 5: Copy and install Python dependencies
# ------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# STEP 6: Copy application source code
# ------------------------------------------------------------
COPY src/ ./src/
COPY models/ ./models/

# ------------------------------------------------------------
# STEP 7: Expose application port
# ------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------
# STEP 8: Run FastAPI application
# ------------------------------------------------------------
# main.py lives in src/, so module path is src.main:app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
