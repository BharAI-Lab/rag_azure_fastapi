FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /webapp

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Create non-root user (safer to run apps)
RUN useradd -m -u 10001 appuser

# Install Python dependencies
COPY app/requirements.txt /webapp/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY app /webapp/app
COPY web /webapp/web

# Run as non-root user
USER appuser

# Start Uvicorn app (keep host 0.0.0.0 so container accepts external connections)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
