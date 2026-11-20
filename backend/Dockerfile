# AuroraStay QnA — RAG Backend image
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + serialized vector database artifacts
COPY app.py config.json chunks.json chunk_embeddings.npy faiss_index.bin ./

# The embedding model is downloaded from HuggingFace on first boot;
# set HUGGINGFACE_HUB_CACHE or mount a volume in production for speed.
ENV PORT=5000
EXPOSE 5000

CMD ["python", "app.py"]
