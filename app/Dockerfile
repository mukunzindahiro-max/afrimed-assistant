FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    GRADIO_SERVER_NAME=0.0.0.0 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

WORKDIR /app

RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt "numpy==1.26.4" "huggingface_hub==0.36.2" "gradio==6.17.3"

COPY . .

EXPOSE 7860
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-7860} python app.py"]
