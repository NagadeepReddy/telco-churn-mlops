# ── Stage 1: Install dependencies ────────────────────────────────
# Separate build stage so the final image is smaller
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system deps needed to compile some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Production image ─────────────────────────────────────
FROM python:3.11-slim AS production

# Security: never run as root in production
RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy FastAPI server code
COPY app/main.py .

# Copy the trained model file
COPY model/churn_model.pkl ./model/churn_model.pkl

# Give the app user ownership
RUN mkdir -p /tmp && chown -R appuser:appgroup /app /tmp

# Switch to non-root user
USER appuser

# FastAPI runs on port 8080
EXPOSE 8080

# Kubernetes health check
# Docker will also use this when running locally
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" \
    || exit 1

# Start FastAPI with uvicorn
# 4 workers = 4 processes handling requests in parallel
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
#     "--workers", "1", \
     "--log-level", "info"]
