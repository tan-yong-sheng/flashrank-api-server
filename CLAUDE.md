# FastAPI Cohere-Compatible Reranker using FlashRank

## Project Overview
Create a FastAPI wrapper that provides a Cohere-compatible reranker API endpoint, leveraging FlashRank's ultra-fast CPU-based reranking models.

## Architecture
```
Cohere Request → FastAPI → FlashRank Format → FlashRank Inference → Relevance Scores → Cohere Response
```

## Key Components

### 1. FastAPI Application
- **Endpoint**: `POST /v2/rerank`
- **Pydantic models** for Cohere API request/response validation
- **Error handling** and proper HTTP status codes

### 2. Request Transformer
- Convert Cohere rerank request format to FlashRank format
- Transform `documents` list to FlashRank `passages` format with metadata
- Handle parameters like `top_n`, `max_tokens_per_doc`

### 3. FlashRank Integration
- Initialize FlashRank `Ranker` with appropriate model
- Create `RerankRequest` objects from transformed input
- Handle different FlashRank models (TinyBERT, MiniLM, T5, etc.)

### 4. Response Processor
- Extract relevance scores from FlashRank results
- Sort documents by relevance score (already done by FlashRank)
- Apply `top_n` filtering if specified

### 5. Cohere Response Formatter
- Transform scored results back to Cohere rerank API format
- Return `RerankResult` objects with `text`, `index`, `relevance_score`
- Apply `top_n` filtering if specified

## Implementation Steps

1. **Setup FastAPI project structure**
   - Create main application file
   - Define Pydantic models for API schemas
   - Setup dependencies and configuration

2. **Implement core rerank endpoint**
   - Create `/v2/rerank` POST endpoint
   - Add request validation and error handling
   - Implement basic response structure

3. **Build request transformation logic**
   - Convert Cohere format to FlashRank RerankRequest format
   - Handle document truncation based on `max_tokens_per_doc`
   - Transform documents list to passages with IDs and metadata

4. **Add FlashRank integration**
   - Initialize FlashRank Ranker with selected model
   - Handle model selection based on request parameters
   - Configure model cache and performance settings

5. **Implement response transformation**
   - Extract scores and rankings from FlashRank results
   - Handle FlashRank's native scoring format
   - Apply result filtering and sorting

6. **Create response transformation**
   - Sort documents by relevance scores
   - Format as Cohere-compatible RerankResult objects
   - Apply top_n filtering

7. **Add configuration and deployment**
   - Environment variables for FlashRank model selection
   - Model caching and memory optimization
   - Health check endpoints and model warmup

## API Specification

### Request Format (Cohere-compatible)
```json
{
  "model": "rerank-v3.5",
  "query": "What is the capital of the United States?",
  "documents": ["Washington D.C. is the capital...", "New York is a major city..."],
  "top_n": 5,
  "max_tokens_per_doc": 4096
}
```

### Response Format (Cohere-compatible)
```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.9109375,
      "document": {
        "text": "Washington D.C. is the capital..."
      }
    }
  ]
}
```

## Dependencies
- **FastAPI**: Web framework
- **Pydantic**: Data validation and serialization
- **FlashRank**: Ultra-fast CPU-based reranking library
- **uvicorn**: ASGI server for deployment

## Configuration
- **FLASHRANK_MODEL**: FlashRank model to use (default: ms-marco-TinyBERT-L-2-v2)
- **FLASHRANK_CACHE_DIR**: Directory for model caching
- **MAX_LENGTH**: Maximum token length for passages (default: 512)
- **MAX_DOCUMENTS**: Maximum number of documents to process
- **DEFAULT_TOP_N**: Default number of results to return

## Available FlashRank Models
- **ms-marco-TinyBERT-L-2-v2**: Default, ultra-lightweight (~4MB)
- **ms-marco-MiniLM-L-12-v2**: Best performance (~34MB)
- **rank-T5-flan**: Best zero-shot performance (~110MB)
- **ms-marco-MultiBERT-L-12**: Multilingual support (~150MB)

This implementation provides a drop-in replacement for Cohere's rerank API while leveraging FlashRank's ultra-fast CPU-based reranking models.