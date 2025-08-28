#!/usr/bin/env python3
"""
Test script for FlashRank Cohere-Compatible Reranker API
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_root_endpoint():
    """Test the root endpoint"""
    print("Testing root endpoint...")
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_list_models():
    """Test the models listing endpoint"""
    print("Testing models endpoint...")
    
    response = requests.get(f"{BASE_URL}/v2/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_basic_rerank():
    """Test basic reranking functionality"""
    print("Testing basic reranking...")
    
    # First get available models
    response = requests.get(f"{BASE_URL}/v2/models")
    if response.status_code != 200:
        print("Failed to get models list")
        return
    
    models_data = response.json()
    available_models = [model["name"] for model in models_data.get("models", [])]
    
    if not available_models:
        print("No models available for testing")
        return
    
    # Use first available model
    model_to_use = available_models[0]
    print(f"Using model: {model_to_use}")
    
    payload = {
        "model": model_to_use,
        "query": "What is the capital of the United States?",
        "documents": [
            "Carson City is the capital city of the American state of Nevada.",
            "Washington, D.C. is the capital of the United States.",
            "New York City is the most populous city in the United States.",
            "Los Angeles is a major city in California.",
            "The United States federal government is headquartered in Washington, D.C."
        ]
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/v2/rerank", json=payload)
    duration = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Duration: {duration:.3f}s")
    
    if response.status_code == 200:
        results = response.json()["results"]
        print(f"Results count: {len(results)}")
        
        for i, result in enumerate(results[:3]):  # Show top 3
            print(f"  {i+1}. Index: {result['index']}, Score: {result['relevance_score']:.4f}")
            print(f"     Text: {result['document']['text'][:80]}...")
    else:
        print(f"Error: {response.text}")
    print()

def test_top_n_filtering():
    """Test top-N filtering"""
    print("Testing top-N filtering...")
    
    # Get available models
    response = requests.get(f"{BASE_URL}/v2/models")
    if response.status_code != 200:
        print("Failed to get models list")
        return
    
    models_data = response.json()
    available_models = [model["name"] for model in models_data.get("models", [])]
    
    if not available_models:
        print("No models available for testing")
        return
    
    model_to_use = available_models[0]
    
    payload = {
        "model": model_to_use,
        "query": "machine learning algorithms",
        "documents": [
            "Neural networks are a type of machine learning algorithm used for pattern recognition.",
            "Linear regression is a statistical method for modeling relationships between variables.",
            "Decision trees are used in machine learning for both classification and regression tasks.",
            "Cooking recipes require precise measurements and timing for best results.",
            "Random forests combine multiple decision trees to improve prediction accuracy.",
            "Support vector machines are powerful algorithms for classification problems.",
            "Weather forecasting uses atmospheric data to predict future conditions."
        ],
        "top_n": 3
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/v2/rerank", json=payload)
    duration = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Duration: {duration:.3f}s")
    
    if response.status_code == 200:
        results = response.json()["results"]
        print(f"Results count: {len(results)} (requested top_n=3)")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. Index: {result['index']}, Score: {result['relevance_score']:.4f}")
            print(f"     Text: {result['document']['text'][:60]}...")
    else:
        print(f"Error: {response.text}")
    print()

def test_different_models():
    """Test different model support"""
    print("Testing different models...")
    
    # First get available models
    response = requests.get(f"{BASE_URL}/v2/models")
    if response.status_code != 200:
        print("    Failed to get models list")
        return
    
    models_data = response.json()
    available_models = [model["name"] for model in models_data.get("models", [])]
    
    if not available_models:
        print("    No models downloaded locally")
        return
    
    print(f"    Available models: {available_models}")
    
    payload_base = {
        "query": "artificial intelligence and machine learning",
        "documents": [
            "Artificial intelligence encompasses machine learning and deep learning techniques.",
            "Weather patterns are influenced by atmospheric pressure and temperature changes.",
            "Machine learning algorithms can be supervised, unsupervised, or reinforcement-based."
        ]
    }
    
    for model in available_models[:2]:  # Test first 2 available models
        print(f"  Testing model: {model}")
        payload = {**payload_base, "model": model}
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/v2/rerank", json=payload)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            results = response.json()["results"]
            top_score = results[0]["relevance_score"] if results else 0
            print(f"    Success - Duration: {duration:.3f}s, Top score: {top_score:.4f}")
        else:
            print(f"    Failed - Status: {response.status_code}, Error: {response.text}")
    print()

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    # Test empty documents
    print("  Testing empty documents...")
    payload = {
        "model": "ms-marco-TinyBERT-L-2-v2",
        "query": "test query",
        "documents": []
    }
    response = requests.post(f"{BASE_URL}/v2/rerank", json=payload)
    print(f"    Status: {response.status_code} (expected 422)")
    
    # Test invalid model
    print("  Testing invalid model...")
    payload = {
        "model": "invalid-model-that-does-not-exist",
        "query": "test query",
        "documents": ["test document"]
    }
    response = requests.post(f"{BASE_URL}/v2/rerank", json=payload)
    print(f"    Status: {response.status_code} (expected 422)")
    if response.status_code != 200:
        print(f"    Error message: {response.json().get('detail', 'No detail')}")
    
    # Test missing query
    print("  Testing missing query...")
    payload = {
        "model": "ms-marco-TinyBERT-L-2-v2",
        "documents": ["test document"]
    }
    response = requests.post(f"{BASE_URL}/v2/rerank", json=payload)
    print(f"    Status: {response.status_code} (expected 422)")
    print()

def test_performance():
    """Test performance with larger document set"""
    print("Testing performance with 50 documents...")
    
    # Get available models
    response = requests.get(f"{BASE_URL}/v2/models")
    if response.status_code != 200:
        print("Failed to get models list")
        return
    
    models_data = response.json()
    available_models = [model["name"] for model in models_data.get("models", [])]
    
    if not available_models:
        print("No models available for testing")
        return
    
    model_to_use = available_models[0]
    
    # Generate test documents
    documents = [
        f"This is test document number {i} about various topics including technology, science, and research." 
        for i in range(50)
    ]
    
    # Add some relevant documents
    documents.extend([
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to process complex data.",
        "Natural language processing helps computers understand and generate human language."
    ])
    
    payload = {
        "model": model_to_use,
        "query": "machine learning and artificial intelligence",
        "documents": documents,
        "top_n": 5
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/v2/rerank", json=payload)
    duration = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Duration: {duration:.3f}s for {len(documents)} documents")
    print(f"Throughput: {len(documents)/duration:.1f} docs/sec")
    
    if response.status_code == 200:
        results = response.json()["results"]
        print(f"Returned {len(results)} results")
        
        # Show top 3 results
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. Score: {result['relevance_score']:.4f}")
            print(f"     Text: {result['document']['text'][:60]}...")
    print()

def main():
    """Run all tests"""
    print("FlashRank Cohere-Compatible Reranker API Tests")
    print("=" * 50)
    
    try:
        test_health_check()
        test_root_endpoint()
        test_list_models()
        test_basic_rerank()
        test_top_n_filtering()
        test_different_models()
        test_error_handling()
        test_performance()
        
        print("All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("Connection failed. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    main()