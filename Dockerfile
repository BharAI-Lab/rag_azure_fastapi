FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # default PORT; ACA can override
    PORT=8000 \
    # make logs more helpful in ACA
    UVICORN_LOG_LEVEL=info

WORKDIR /webapp

# Install minimal system deps first (layer caching)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user & group
RUN useradd -m -u 10001 appuser

# Install Python deps
# (Make sure requirements.txt includes: fastapi, uvicorn, any Azure libs you need)
COPY app/requirements.txt /webapp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /webapp/requirements.txt

# Copy application code
COPY app /webapp/app
COPY web /webapp/web

# Ensure the runtime user can read/write if your app writes anything under /webapp
RUN chown -R appuser:appuser /webapp

USER appuser

# Document the container port (ingress target in ACA should be 8000)
EXPOSE 8000

# Start the API server; honor $PORT if ACA provides it
# (use sh -c to allow env expansion)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
