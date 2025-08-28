# FlashRank Cohere-Compatible Reranker

Ultra-fast CPU-based document reranking API that provides Cohere-compatible endpoints using FlashRank.

## Features

- 🚀 **Ultra-fast**: CPU-based inference with models from 4MB to 4GB
- 🔗 **Cohere-compatible**: Drop-in replacement for Cohere's `/v2/rerank` endpoint
- 🎯 **Multiple models**: TinyBERT, MiniLM, T5-Flan, MultiBERT, and Zephyr 7B
- 📦 **Lightweight**: No GPU required, runs anywhere
- 🌐 **Production-ready**: CORS support, health checks, error handling

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install directly
pip install fastapi uvicorn flashrank
```

### Run the Server

```bash
# Start the server
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment (Recommended)

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build and run manually
docker build -t flashrank-api .
docker run -p 8000:8000 flashrank-api
```

### Test the API

```bash
curl -X POST "http://localhost:8000/v2/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ms-marco-TinyBERT-L-2-v2",
    "query": "What is the capital of the United States?",
    "documents": [
      "Carson City is the capital city of Nevada.",
      "Washington, D.C. is the capital of the United States.",
      "New York City is the most populous city in the United States."
    ],
    "top_n": 2
  }'
```

## API Endpoints

### `POST /v2/rerank`
Rerank documents based on relevance to a query (Cohere-compatible).

### `GET /health`
Health check endpoint.

### `GET /v2/models`
List available reranking models.

### `GET /`
Root endpoint with service information.

## Model Support

| FlashRank Model | Size | Performance |
|-----------------|------|-------------|
| `ms-marco-TinyBERT-L-2-v2` (default) | ~4MB | Ultra-fast |
| `ms-marco-MiniLM-L-12-v2` | ~34MB | Best accuracy |
| `rank-T5-flan` | ~110MB | Best zero-shot |
| `ms-marco-MultiBERT-L-12` | ~150MB | Multilingual |
| `ce-esci-MiniLM-L12-v2` | ~34MB | E-commerce |
| `rank_zephyr_7b_v1_full` | ~4GB | Maximum accuracy |
| `miniReranker_arabic_v1` | ~34MB | Arabic language |

## Configuration

Set environment variables:

```bash
export MAX_DOCUMENTS=1000          # Maximum documents per request
export DEFAULT_TOP_N=100           # Default number of results
export MAX_LENGTH=512              # Maximum token length
export FLASHRANK_CACHE_DIR=/tmp    # Model cache directory
export HOST=0.0.0.0               # Server host
export PORT=8000                  # Server port
export LOG_LEVEL=info             # Logging level
```

## Documentation

- **API Reference**: `docs/cohere-api-reference.md`
- **FlashRank Usage**: `docs/flashrank-usage.md`
- **Interactive docs**: http://localhost:8000/docs

## Performance

- **TinyBERT**: 1000+ docs/sec, ~50MB RAM
- **MiniLM**: 500+ docs/sec, ~200MB RAM
- **T5-Flan**: 200+ docs/sec, ~500MB RAM

Perfect for production APIs requiring high throughput and low latency.