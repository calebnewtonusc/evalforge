FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 evalforge && chown -R evalforge:evalforge /app
USER evalforge

EXPOSE 9000

ENV EVALFORGE_API_HOST=0.0.0.0
ENV EVALFORGE_API_PORT=9000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

CMD ["uvicorn", "deploy.api_server:app", "--host", "0.0.0.0", "--port", "9000"]
