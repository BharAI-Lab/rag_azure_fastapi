FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /webapp

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Create a non-root user (security best practice)
RUN useradd -m -u 10001 appuser

# Install deps
COPY app/requirements.txt /webapp/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Copy code
COPY app /webapp/app
COPY web /webapp/web

EXPOSE 8000

# Drop privileges
USER appuser

ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
