# API Examples - Multi-Agent RAG System

This document provides comprehensive examples of how to interact with the Multi-Agent RAG System API.

## Table of Contents

- [Basic Usage](#basic-usage)
- [cURL Examples](#curl-examples)
- [Python Examples](#python-examples)
- [JavaScript Examples](#javascript-examples)
- [Session Management](#session-management)
- [Different Question Types](#different-question-types)
- [Error Handling](#error-handling)

## Basic Usage

The API runs on `http://localhost:8000` by default. All examples assume this base URL.

## cURL Examples

### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "pdf_service": true,
  "papers_loaded": 4,
  "agent_system": true
}
```

### Ask a Question (Auto-Session)

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What papers are available in the database?"
  }'
```

**Response:**
```json
{
  "response": "The database contains the following papers:\n\nPaper: Chang and Fosler-Lussier - 2023\nTitle: How to Prompt LLMs for Text-to-SQL: A Study in Zero-shot, Few-shot, and Fine-tuning Approaches\nPages: 12\n\nPaper: Zhang et al. - 2024\nTitle: Benchmarking the Text-to-SQL Capability of Large Language Models\nPages: 45\n\n...",
  "type": "answer",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "search_strategy": "pdf",
  "reasoning": "Clarification agent approved query | Router agent decided: PDF | PDF research agent activated",
  "confidence": 0.0
}
```

### Create Session Explicitly

```bash
curl -X POST "http://localhost:8000/session"
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440001",
  "message": "Session created successfully"
}
```

### Ask Question with Existing Session

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What methodology did Zhang et al. use for benchmarking?",
    "session_id": "550e8400-e29b-41d4-a716-446655440001"
  }'
```

### Get Session History

```bash
curl -X GET "http://localhost:8000/session/550e8400-e29b-41d4-a716-446655440001/history"
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440001",
  "history": [
    "What methodology did Zhang et al. use for benchmarking?",
    "Zhang et al. (2024) used a comprehensive benchmarking methodology..."
  ]
}
```

### Clear Session

```bash
curl -X DELETE "http://localhost:8000/session/550e8400-e29b-41d4-a716-446655440001"
```

### Re-ingest PDFs

```bash
curl -X POST "http://localhost:8000/ingest"
```

## Python Examples

### Using requests library

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Health check
def check_health():
    response = requests.get(f"{BASE_URL}/health")
    return response.json()

# Ask a question
def ask_question(question, session_id=None):
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    
    response = requests.post(f"{BASE_URL}/ask", json=payload)
    return response.json()

# Create session
def create_session():
    response = requests.post(f"{BASE_URL}/session")
    return response.json()["session_id"]

# Example usage
if __name__ == "__main__":
    # Check if service is healthy
    health = check_health()
    print(f"Service status: {health['status']}")
    
    # Create a session
    session_id = create_session()
    print(f"Created session: {session_id}")
    
    # Ask questions
    questions = [
        "What papers are available?",
        "What are the main findings in the Zhang et al. paper?",
        "How do the accuracy results compare across papers?",
        "What is the latest news about OpenAI?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = ask_question(question, session_id)
        print(f"A: {result['response'][:200]}...")
        print(f"Strategy: {result.get('search_strategy', 'unknown')}")
```

### Async Python Example

```python
import asyncio
import aiohttp

async def ask_question_async(question, session_id=None):
    async with aiohttp.ClientSession() as session:
        payload = {"question": question}
        if session_id:
            payload["session_id"] = session_id
        
        async with session.post(
            "http://localhost:8000/ask", 
            json=payload
        ) as response:
            return await response.json()

# Usage
async def main():
    result = await ask_question_async("What papers are available?")
    print(result['response'])

asyncio.run(main())
```

## JavaScript Examples

### Using fetch API

```javascript
const BASE_URL = 'http://localhost:8000';

// Health check
async function checkHealth() {
    const response = await fetch(`${BASE_URL}/health`);
    return await response.json();
}

// Ask a question
async function askQuestion(question, sessionId = null) {
    const payload = { question };
    if (sessionId) {
        payload.session_id = sessionId;
    }
    
    const response = await fetch(`${BASE_URL}/ask`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
    });
    
    return await response.json();
}

// Create session
async function createSession() {
    const response = await fetch(`${BASE_URL}/session`, {
        method: 'POST'
    });
    const data = await response.json();
    return data.session_id;
}

// Example usage
async function main() {
    try {
        // Check health
        const health = await checkHealth();
        console.log('Service status:', health.status);
        
        // Create session
        const sessionId = await createSession();
        console.log('Session created:', sessionId);
        
        // Ask question
        const result = await askQuestion(
            "What are the main contributions of the Zhang et al. paper?",
            sessionId
        );
        
        console.log('Response:', result.response);
        console.log('Strategy:', result.search_strategy);
        
    } catch (error) {
        console.error('Error:', error);
    }
}

main();
```

## Different Question Types

### PDF-focused Questions

```bash
# List papers
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What papers are available?"}'

# Specific paper search
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What methodology did Zhang et al. use?"}'

# Comparative analysis
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare the accuracy results across all papers"}'

# Technical details
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What prompt templates gave the best results for Spider dataset?"}'
```

### Web-focused Questions

```bash
# Current events
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What has OpenAI released this month?"}'

# Recent developments
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Latest news about large language models"}'
```

### Hybrid Questions (Both PDF and Web)

```bash
# Academic vs current state
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do the results in these papers compare to current state-of-the-art?"}'
```

## Error Handling

### Handling Errors in Python

```python
import requests

def safe_ask_question(question, session_id=None):
    try:
        payload = {"question": question}
        if session_id:
            payload["session_id"] = session_id
        
        response = requests.post(
            "http://localhost:8000/ask", 
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            return {"error": "Service not ready, please try again"}
        elif response.status_code == 400:
            return {"error": f"Bad request: {response.json().get('detail', 'Unknown error')}"}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - question might be complex"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to service"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Usage
result = safe_ask_question("What papers are available?")
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Response: {result['response']}")
```

## Testing Your Deployment

After deployment, you can run the included test script:

```bash
python test_api.py
```

Or manually test the key endpoints:

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Simple question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What papers are available?"}'

# 3. Check API documentation
# Visit: http://localhost:8000/docs
``` 