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

EXPOSE 1370
EXPOSE 1371
EXPOSE 1372
EXPOSE 1373
EXPOSE 1374
EXPOSE 1375
EXPOSE 1376
EXPOSE 1377
EXPOSE 1378
EXPOSE 1379
EXPOSE 1380

CMD ["python", "app.py", "--model", "Qwen/Qwen2.5-1.5B-Instruct", "--tensor_parallel_size", "1", "--gpu_memory_utilization", "0.92", "--max_model_len", "2048", "--port", "1370"]