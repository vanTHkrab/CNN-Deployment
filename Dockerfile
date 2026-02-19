FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/docker \
    PATH="/home/docker/.local/bin:$PATH"

RUN useradd -m docker \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Use lightweight production deps (tflite-runtime, not tensorflow)
COPY requirements.prod.txt .

USER docker

RUN pip install --upgrade pip \
    && pip install --user -r requirements.prod.txt

# Copy source code and TFLite models only (.keras excluded via .dockerignore)
COPY --chown=docker:docker src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
