FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install huggingface_hub vllm fastapi uvicorn pynvml

WORKDIR /usr/src/app

COPY . /usr/src/app

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

ENTRYPOINT ["uvicorn", "app:app"]


