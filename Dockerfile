FROM python:3.12-slim

WORKDIR /usr/src/app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install huggingface_hub vllm fastapi uvicorn pynvml

COPY . .

ENTRYPOINT ["uvicorn", "app:app"]