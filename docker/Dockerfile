FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install \
      --no-cache-dir \
      --timeout 120 \
      --retries 20 \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
      --trusted-host pypi.tuna.tsinghua.edu.cn \
      torch==2.3.1

RUN python -m pip install -r /tmp/requirements.txt

COPY . /app

CMD ["bash"]
