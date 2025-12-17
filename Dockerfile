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

# Install system dependencies required by some Python packages (e.g., LightGBM)
# libgomp1 provides libgomp.so.1 used by LightGBM wheels built with GCC
RUN apt-get update \
	&& apt-get install -y --no-install-recommends libgomp1 \
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
COPY api/ ./api/
COPY models/ ./models/
COPY data/ ./data/

# ------------------------------------------------------------
# STEP 7: Expose application port
# ------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------
# STEP 8: Run FastAPI application
# ------------------------------------------------------------
# main FastAPI app entrypoint is `api.main:app`
# In development you may want to use --reload; for production omit it.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
