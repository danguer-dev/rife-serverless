FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    wget \
    unzip \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Descargar rife-ncnn-vulkan oficial con todos los pesos incluidos (rife-v4.6)
RUN wget -q https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip -O rife.zip && \
    unzip -q rife.zip && \
    mv rife-ncnn-vulkan-20221029-ubuntu/* /app/ && \
    chmod +x /app/rife-ncnn-vulkan && \
    rm -rf rife.zip rife-ncnn-vulkan-20221029-ubuntu

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
