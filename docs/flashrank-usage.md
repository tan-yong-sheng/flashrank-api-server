# FlashRank Integration Guide

## Overview

FlashRank is an ultra-lightweight, CPU-based reranking library that provides state-of-the-art document reranking capabilities. This guide covers how to use FlashRank for building high-performance reranking systems.

## Installation

### Basic Installation (Pairwise Rerankers)
```bash
pip install flashrank
```

### Advanced Installation (Listwise Rerankers)
```bash
pip install flashrank[listwise]
```

## Quick Start

### Basic Usage
```python
from flashrank import Ranker, RerankRequest

# Initialize with default model (TinyBERT, ~4MB)
ranker = Ranker(max_length=512)

# Prepare your data
query = "How to speedup LLMs?"
passages = [
    {
        "id": 1,
        "text": "Introduce lookahead decoding: a parallel decoding algo to accelerate LLM inference",
        "meta": {"source": "paper_1"}
    },
    {
        "id": 2,
        "text": "LLM inference efficiency will be crucial for both industry and academia",
        "meta": {"source": "paper_2"}
    },
    {
        "id": 3,
        "text": "vLLM is a fast and easy-to-use library for LLM inference and serving",
        "meta": {"source": "paper_3"}
    }
]

# Create rerank request
rerank_request = RerankRequest(query=query, passages=passages)

# Get ranked results
results = ranker.rerank(rerank_request)
print(results)
```

### Output Format
```python
[
    {
        "id": 1,
        "text": "Introduce lookahead decoding: a parallel decoding algo to accelerate LLM inference",
        "meta": {"source": "paper_1"},
        "score": 0.8934567
    },
    {
        "id": 3,
        "text": "vLLM is a fast and easy-to-use library for LLM inference and serving",
        "meta": {"source": "paper_3"},
        "score": 0.7823456
    },
    {
        "id": 2,
        "text": "LLM inference efficiency will be crucial for both industry and academia",
        "meta": {"source": "paper_2"},
        "score": 0.6712345
    }
]
```

## Available Models

### 1. TinyBERT (Default - Ultra-Lightweight)
```python
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", max_length=512)
```
- **Size**: ~4MB
- **Speed**: Fastest
- **Use Case**: High-throughput applications, serverless deployments
- **Performance**: Good for most general ranking tasks

### 2. MiniLM (Best Performance)
```python
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", max_length=512)
```
- **Size**: ~34MB
- **Speed**: Fast
- **Use Case**: When accuracy is priority over size
- **Performance**: Best cross-encoder performance

### 3. T5-Flan (Best Zero-shot)
```python
ranker = Ranker(model_name="rank-T5-flan", max_length=512)
```
- **Size**: ~110MB
- **Speed**: Moderate
- **Use Case**: Out-of-domain data, zero-shot scenarios
- **Performance**: Best for diverse/unseen domains

### 4. MultiBERT (Multilingual)
```python
ranker = Ranker(model_name="ms-marco-MultiBERT-L-12", max_length=512)
```
- **Size**: ~150MB
- **Speed**: Moderate
- **Use Case**: 100+ languages support
- **Performance**: Good for non-English content

### 5. Zephyr 7B (LLM-based)
```python
ranker = Ranker(model_name="rank_zephyr_7b_v1_full", max_length=1024)
```
- **Size**: ~4GB (4-bit quantized)
- **Speed**: Slower
- **Use Case**: Maximum accuracy scenarios
- **Performance**: State-of-the-art listwise ranking

## Configuration Options

### Cache Directory
```python
# Specify custom cache directory (useful for serverless)
ranker = Ranker(
    model_name="ms-marco-MiniLM-L-12-v2",
    cache_dir="/opt/models"  # For AWS Lambda
)
```

### Max Length Optimization
```python
# Optimize for your content length
ranker = Ranker(
    model_name="ms-marco-TinyBERT-L-2-v2",
    max_length=128  # For short passages (~100 tokens)
)

ranker = Ranker(
    model_name="ms-marco-MiniLM-L-12-v2", 
    max_length=512  # For longer documents
)
```

## Data Format Requirements

### Passage Structure
```python
passage = {
    "id": 1,                    # Required: unique identifier
    "text": "Document content", # Required: text to rank
    "meta": {                   # Optional: additional metadata
        "title": "Document Title",
        "source": "database",
        "timestamp": "2024-01-01"
    }
}
```

### Simple String Format (Auto-converted)
```python
# FlashRank can also handle simple strings
passages = [
    "First document text",
    "Second document text", 
    "Third document text"
]

# Automatically converts to:
# [{"id": 0, "text": "First document text"}, ...]
```

## Performance Optimization

### 1. Token Length Estimation
```python
import tiktoken

def estimate_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Optimize max_length based on your data
max_tokens = max(estimate_tokens(query) + estimate_tokens(passage["text"]) 
                for passage in passages)
optimal_max_length = min(max_tokens + 50, 512)  # Add buffer, cap at 512

ranker = Ranker(max_length=optimal_max_length)
```

### 2. Batch Processing
```python
def process_large_dataset(query, all_passages, batch_size=50):
    results = []
    
    for i in range(0, len(all_passages), batch_size):
        batch = all_passages[i:i + batch_size]
        request = RerankRequest(query=query, passages=batch)
        batch_results = ranker.rerank(request)
        results.extend(batch_results)
    
    # Re-sort globally
    return sorted(results, key=lambda x: x["score"], reverse=True)
```

### 3. Memory Management
```python
import gc

def rerank_with_cleanup(query, passages):
    ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)
    
    # Cleanup for memory-constrained environments
    del ranker
    gc.collect()
    
    return results
```

## Integration Patterns

### 1. Search Pipeline Integration
```python
class SearchPipeline:
    def __init__(self):
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    
    def search_and_rerank(self, query, initial_results, top_k=10):
        # Convert search results to FlashRank format
        passages = [
            {
                "id": result["id"],
                "text": result["content"],
                "meta": {"score": result["search_score"]}
            }
            for result in initial_results
        ]
        
        # Rerank
        request = RerankRequest(query=query, passages=passages)
        reranked = self.ranker.rerank(request)
        
        return reranked[:top_k]
```

### 2. RAG (Retrieval-Augmented Generation)
```python
class RAGPipeline:
    def __init__(self):
        self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
    
    def retrieve_and_rerank(self, question, retrieved_docs, top_k=5):
        passages = [
            {"id": i, "text": doc, "meta": {}}
            for i, doc in enumerate(retrieved_docs)
        ]
        
        request = RerankRequest(query=question, passages=passages)
        reranked = self.ranker.rerank(request)
        
        # Return top-k most relevant for LLM context
        return [item["text"] for item in reranked[:top_k]]
```

### 3. Serverless Deployment (AWS Lambda)
```python
import os
import json
from flashrank import Ranker, RerankRequest

# Global variable for model reuse across invocations
ranker = None

def lambda_handler(event, context):
    global ranker
    
    # Initialize once (cold start)
    if ranker is None:
        ranker = Ranker(
            model_name="ms-marco-TinyBERT-L-2-v2",
            cache_dir="/tmp/flashrank"  # Lambda tmp directory
        )
    
    # Parse request
    body = json.loads(event["body"])
    query = body["query"]
    documents = body["documents"]
    
    # Convert to FlashRank format
    passages = [
        {"id": i, "text": doc, "meta": {}}
        for i, doc in enumerate(documents)
    ]
    
    # Rerank
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)
    
    return {
        "statusCode": 200,
        "body": json.dumps({"results": results})
    }
```

## Benchmarking and Performance

### Speed Comparison (Approximate)
| Model | Size | Speed (docs/sec) | Use Case |
|-------|------|-----------------|----------|
| TinyBERT | 4MB | 1000+ | High throughput |
| MiniLM | 34MB | 500+ | Balanced |
| T5-Flan | 110MB | 200+ | Zero-shot |
| MultiBERT | 150MB | 150+ | Multilingual |
| Zephyr 7B | 4GB | 50+ | Maximum accuracy |

### Memory Usage
- **TinyBERT**: ~50MB RAM
- **MiniLM**: ~200MB RAM  
- **T5-Flan**: ~500MB RAM
- **MultiBERT**: ~800MB RAM
- **Zephyr 7B**: ~8GB RAM

## Best Practices

1. **Model Selection**: 
   - Use TinyBERT for production APIs with high QPS
   - Use MiniLM when accuracy is critical
   - Use MultiBERT only for non-English content

2. **Performance Tuning**:
   - Set `max_length` to just above your longest document
   - Process 10-100 documents per batch for optimal throughput
   - Cache ranker instances in production

3. **Memory Management**:
   - Use `/tmp` for caching in serverless environments
   - Consider model cleanup for memory-constrained systems
   - Monitor memory usage in production

4. **Error Handling**:
   - Handle empty document lists gracefully
   - Validate input text lengths
   - Implement fallback ranking strategies

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   ```python
   # Use smaller model or reduce max_length
   ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", max_length=256)
   ```

2. **Slow Performance**:
   ```python
   # Optimize max_length for your content
   ranker = Ranker(max_length=128)  # Instead of default 512
   ```

3. **Model Download Issues**:
   ```python
   # Specify custom cache directory
   ranker = Ranker(cache_dir="/path/to/writable/directory")
   ```

4. **Large Document Handling**:
   ```python
   # Truncate documents before ranking
   def truncate_text(text, max_tokens=400):
       words = text.split()
       return " ".join(words[:max_tokens])
   
   passages = [
       {"id": i, "text": truncate_text(doc), "meta": {}}
       for i, doc in enumerate(documents)
   ]
   ```