# Hugging Face Spaces deployment image for the arXiv RAG FastAPI service.
FROM python:3.12-slim

# System libraries:
#   - libgomp1: required at runtime by onnxruntime (the FlashRank reranker)
#   - build-essential: lets pip compile any dependency lacking a prebuilt wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: run as a non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Python deps first so this layer is cached across code changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project: source code + prebuilt chroma_db + BM25 pickle.
COPY --chown=user . .

# HF routes traffic to the port declared as app_port in the README frontmatter.
EXPOSE 7860
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
