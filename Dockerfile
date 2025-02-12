# Устанавливаем CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
RUN echo "cuda установлен успешно"

# Устанавливаем временную зону по умолчанию
ENV DEBIAN_FRONTEND=noninteractive
RUN rm -f /etc/apt/sources.list.d/cuda* \
    && apt-get update --allow-releaseinfo-change \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y tzdata \
    python3.11 python3.11-dev python3.11-venv curl \
    libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/cache/apt /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py

# Создаем ссылку на python3.11 как python и python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app
COPY . /app

# Устанавливаем PyTorch и зависимости
#COPY wheels/ /app/wheels/

RUN pip install --no-cache-dir /app/wheels/torch-2.5.0+cu121-cp311-cp311-linux_x86_64.whl
RUN pip install --no-cache-dir /app/wheels/torchaudio-2.5.0+cu121-cp311-cp311-linux_x86_64.whl
RUN pip install --no-cache-dir /app/wheels/torchvision-0.20.1+cu121-cp311-cp311-linux_x86_64.whl

RUN pip install thop \
        -r requirements.txt  \
    && rm -rf ~/.cache/pip

# Копируем проект
#COPY requirements.txt /app/
#RUN pip install -r requirements.txt && rm -rf ~/.cache/pip

# Открытие порта
EXPOSE 8080

# Команда для запуска вашего приложения
CMD ["python3", "api.py"]