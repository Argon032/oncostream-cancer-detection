FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir kaggle

# Copy project files
COPY . .

# Download model checkpoints from Kaggle into the correct paths
RUN --mount=type=secret,id=KAGGLE_USERNAME \
    --mount=type=secret,id=KAGGLE_KEY \
    mkdir -p /root/.kaggle && \
    echo "{\"username\":\"$(cat /run/secrets/KAGGLE_USERNAME)\",\"key\":\"$(cat /run/secrets/KAGGLE_KEY)\"}" > /root/.kaggle/kaggle.json && \
    chmod 600 /root/.kaggle/kaggle.json && \
    mkdir -p results/brain results/breast && \
    kaggle datasets download -d argon03/oncostream-model-checkpoints --unzip -p /tmp/ckpts && \
    find /tmp/ckpts -type f && \
    cp /tmp/ckpts/brain/vit_best.pth results/brain/vit_best.pth && \
    cp /tmp/ckpts/breast/resnet50_best.pth results/breast/resnet50_best.pth && \
    rm -rf /tmp/ckpts

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]