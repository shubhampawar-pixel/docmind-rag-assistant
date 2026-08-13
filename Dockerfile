FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=7860 \
    LLM_PROVIDER=gemini

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir chroma-hnswlib --only-binary :all: && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure data directories exist
RUN mkdir -p data/documents data/chroma_db data/bm25_cache

# Expose Hugging Face default port 7860
EXPOSE 7860

# Run Streamlit on port 7860
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
