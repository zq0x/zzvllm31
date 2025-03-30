FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install huggingface_hub vllm fastapi uvicorn pynvml

WORKDIR /app

COPY app.py /app/app.py

ENTRYPOINT ["uvicorn", "app:app"]
