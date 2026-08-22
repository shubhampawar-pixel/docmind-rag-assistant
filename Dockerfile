FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000 \
    LLM_PROVIDER=gemini

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir chroma-hnswlib --only-binary :all: && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure data directories exist
RUN mkdir -p data/documents data/chroma_db data/bm25_cache

# Expose FastAPI port 8000 and Streamlit port 8501
EXPOSE 8000 8501

# Default command: Start FastAPI REST service
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
