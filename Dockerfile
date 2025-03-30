FROM python:3.9-slim-buster

WORKDIR /usr/src/app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install huggingface_hub vllm fastapi uvicorn pynvml

COPY . .

CMD ["python", "app.py", "--model", "Qwen/Qwen2.5-1.5B-Instruct", "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.92", "--max-model-len", "2048", "--port", "1370"]