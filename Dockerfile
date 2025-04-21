FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git ffmpeg libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/DeepLearningAtariAgent 

COPY . .

ENV PYTHONPATH=/workspace/DeepLearningAtariAgent

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements2.txt

ENTRYPOINT ["python", "train.py"]

CMD ["--episodes", "1000", "--device", "cuda"]
