# Cohere Rerank API Reference

## Overview

The `/v2/rerank` endpoint provides document reranking functionality compatible with Cohere's API specification. It takes a query and a list of documents, then returns them ranked by relevance.

## Endpoint

```
POST /v2/rerank
```

## Request Format

### Headers
```bash
Content-Type: application/json
```

### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | The reranking model to use (e.g., "rerank-v3.5") |
| `query` | string | Yes | The search query to rank documents against |
| `documents` | array[string] | Yes | List of document texts to rerank |
| `top_n` | integer | No | Maximum number of documents to return (default: all) |
| `max_tokens_per_doc` | integer | No | Maximum tokens per document (default: 4096) |

### Example Request Body
```json
{
  "model": "rerank-v3.5",
  "query": "What is the capital of the United States?",
  "documents": [
    "Carson City is the capital city of the American state of Nevada.",
    "Washington, D.C. is the capital of the United States.",
    "New York City is the most populous city in the United States.",
    "Los Angeles is a major city in California.",
    "The United States federal government is headquartered in Washington, D.C."
  ],
  "top_n": 3,
  "max_tokens_per_doc": 512
}
```

## Response Format

### Success Response (200 OK)

```json
{
  "results": [
    {
      "index": 1,
      "relevance_score": 0.9108734,
      "document": {
        "text": "Washington, D.C. is the capital of the United States."
      }
    },
    {
      "index": 4,
      "relevance_score": 0.8567432,
      "document": {
        "text": "The United States federal government is headquartered in Washington, D.C."
      }
    },
    {
      "index": 2,
      "relevance_score": 0.2341567,
      "document": {
        "text": "New York City is the most populous city in the United States."
      }
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | Array of ranked document results |
| `results[].index` | integer | Original index of the document in the input array |
| `results[].relevance_score` | float | Relevance score between 0 and 1 |
| `results[].document.text` | string | The original document text |

## cURL Examples

### Basic Reranking Request
```bash
curl -X POST "http://localhost:8000/v2/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank-v3.5",
    "query": "What is the capital of the United States?",
    "documents": [
      "Carson City is the capital city of the American state of Nevada.",
      "Washington, D.C. is the capital of the United States.",
      "New York City is the most populous city in the United States."
    ]
  }'
```

### With Top-N Filtering
```bash
curl -X POST "http://localhost:8000/v2/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank-v3.5",
    "query": "machine learning algorithms",
    "documents": [
      "Neural networks are a type of machine learning algorithm.",
      "Linear regression is a statistical method.",
      "Decision trees are used in machine learning for classification.",
      "Cooking recipes require precise measurements.",
      "Random forests combine multiple decision trees."
    ],
    "top_n": 3
  }'
```

### With Token Limit
```bash
curl -X POST "http://localhost:8000/v2/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank-v3.5",
    "query": "artificial intelligence",
    "documents": [
      "Artificial intelligence (AI) is intelligence demonstrated by machines...",
      "Machine learning is a method of data analysis that automates analytical model building..."
    ],
    "max_tokens_per_doc": 256,
    "top_n": 5
  }'
```

### Complex Query Example
```bash
curl -X POST "http://localhost:8000/v2/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank-v3.5",
    "query": "How to optimize database performance for large datasets",
    "documents": [
      "Database indexing is crucial for query performance. Proper indexes can reduce query time from seconds to milliseconds.",
      "Normalization reduces data redundancy but can impact performance through joins.",
      "Caching frequently accessed data in memory can significantly improve response times.",
      "Horizontal partitioning divides large tables across multiple databases.",
      "Weather forecast shows rain tomorrow.",
      "Query optimization involves analyzing execution plans and identifying bottlenecks.",
      "Connection pooling helps manage database connections efficiently."
    ],
    "top_n": 4,
    "max_tokens_per_doc": 512
  }'
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request: missing required field 'query'"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "documents"],
      "msg": "ensure this value has at least 1 items",
      "type": "value_error.list.min_items"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error during reranking"
}
```

## Model Support

The following models are supported:

- `rerank-v3.5` (default): Maps to FlashRank's TinyBERT model
- `rerank-english-v3.0`: Maps to FlashRank's MiniLM model
- `rerank-multilingual-v3.0`: Maps to FlashRank's MultiBERT model

## Rate Limits

- No built-in rate limiting (depends on server configuration)
- CPU-based inference allows for high throughput
- Memory usage scales with document count and length

## Best Practices

1. **Document Length**: Keep documents under 512 tokens for optimal performance
2. **Batch Size**: Process 10-100 documents per request for best balance of speed and accuracy
3. **Query Quality**: Use specific, descriptive queries for better ranking results
4. **Top-N**: Use `top_n` parameter to limit results and improve response time
5. **Model Selection**: Use TinyBERT for speed, MiniLM for accuracy, MultiBERT for multilingual content