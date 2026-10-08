# The live demo as a Hugging Face Space (Docker SDK: Spaces' built-in Streamlit SDK is
# deprecated). deploy/push_space.py uploads this file with app.py, requirements.txt, the
# Space README and the project's Python packages (triage, retrieval, ingestion, evals).
FROM python:3.13-slim

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# CPU-only torch first (the default Linux wheel adds ~2 GB of CUDA libraries).
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image, so a cold start doesn't download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY --chown=user . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
