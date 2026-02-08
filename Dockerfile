FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/docker \
    PATH="/home/docker/.local/bin:$PATH"

RUN useradd -m docker \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

USER docker

RUN pip install --upgrade pip \
    && pip install --user -r requirements.txt
