FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clonar repo base de Practical-RIFE y descargar pesos pre-entrenados
RUN git clone https://github.com/hzwer/Practical-RIFE.git . && \
    mkdir -p train_log

# Descarga de pesos HDv3 / v4 (o copiarlos localmente si los tienes descargados)
# RUN curl -L -o train_log/flownet.pkl <URL_DIRECTA_DE_PESOS_RIFE>

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]