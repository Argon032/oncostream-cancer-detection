FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir kaggle

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "mkdir -p /root/.kaggle results/brain results/breast && python3 -c \"import json,os; json.dump({'username':os.environ['KAGGLE_USERNAME'],'key':os.environ['KAGGLE_KEY']},open('/root/.kaggle/kaggle.json','w'))\" && chmod 600 /root/.kaggle/kaggle.json && kaggle datasets download -d argon03/oncostream-model-checkpoints --unzip -p /tmp/ckpts && mv /tmp/ckpts/PTH_history/brain/vit_best.pth results/brain/ && mv /tmp/ckpts/PTH_history/breast/resnet50_best.pth results/breast/ && uvicorn api.main:app --host 0.0.0.0 --port 7860"]