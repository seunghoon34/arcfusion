# Multi-Agent RAG System

A sophisticated multi-agent system for research paper analysis and web search, built with FastAPI, LangChain, and LangGraph.

## Features

- **Multi-Agent Architecture**: 6 specialized agents working together:
  - Clarification Agent: Detects ambiguous queries
  - Router Agent: Decides search strategy (PDF/Web/Both)
  - PDF Research Agent: Searches academic papers
  - Web Research Agent: Searches current information
  - Tool Execution Agent: Executes research tools
  - Response Synthesis Agent: Creates final answers

- **PDF Processing**: Automatic ingestion and indexing of research papers
- **Vector Search**: ChromaDB-powered semantic search
- **Web Search**: Integration with Google Serper API
- **Session Management**: Conversation history and context
- **REST API**: FastAPI-based endpoints for easy integration

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key
- Google Serper API key (optional, for web search)

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional but recommended for web search
SERPER_API_KEY=your_serper_api_key_here

# Application Settings (optional)
PAPERS_FOLDER=papers
PERSIST_DIRECTORY=/app/vectorstore
```

## Installation & Setup

### Option 1: Docker Compose (Recommended)

1. Clone the repository and navigate to the project directory
2. Create your `.env` file with the required API keys
3. Ensure your PDF files are in the `papers/` directory
4. Run the application:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

### Option 2: Local Development

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Create your `.env` file with the required API keys
3. Ensure your PDF files are in the `papers/` directory
4. Run the application:

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Root Endpoint
- `GET /` - API information and available endpoints

### Core Functionality
- `POST /ask` - Ask a question to the multi-agent system
- `POST /session` - Create a new conversation session
- `GET /session/{session_id}/history` - Get session history
- `DELETE /session/{session_id}` - Clear session history

### Management
- `POST /ingest` - Re-ingest PDFs (useful when adding new papers)
- `GET /health` - Health check endpoint

## Usage Examples

### Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings in the Zhang et al. paper?",
    "session_id": "optional-session-id"
  }'
```

### Create a Session

```bash
curl -X POST "http://localhost:8000/session"
```

### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

## API Documentation

Once the application is running, you can access:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **API Examples**: See [API_EXAMPLES.md](API_EXAMPLES.md) for comprehensive usage examples

## Adding New Papers

1. Place PDF files in the `papers/` directory
2. Call the `/ingest` endpoint to re-process the papers:

```bash
curl -X POST "http://localhost:8000/ingest"
```

Or restart the application to automatically detect and process new papers.

## File Structure

```
arcfusion/
├── main.py                 # FastAPI application
├── ingest_pdf.py          # PDF ingestion service
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Docker Compose setup
├── test_api.py           # API testing script
├── README.md            # Setup and deployment guide
├── API_EXAMPLES.md      # Comprehensive API usage examples
├── papers/              # PDF research papers
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
└── .env                 # Environment variables (create this)
```

## Architecture

The system uses a multi-agent architecture built with LangGraph:

1. **Clarification Agent**: Analyzes queries for ambiguity
2. **Router Agent**: Determines the best search strategy
3. **PDF Research Agent**: Searches through ingested papers
4. **Web Research Agent**: Searches current web information
5. **Tool Execution Agent**: Executes the appropriate tools
6. **Response Synthesis Agent**: Combines results into final response

## Configuration

### PDF Processing
- Papers are automatically chunked using RecursiveCharacterTextSplitter
- Chunks are embedded using OpenAI's text-embedding-3-small model
- Stored in ChromaDB for efficient similarity search

### Search Strategies
- **PDF**: Search only in research papers
- **Web**: Search only current web information  
- **Both**: Search papers first, then web for comprehensive answers

## Troubleshooting

### Common Issues

1. **PDF files not found**: Ensure PDFs are in the `papers/` directory
2. **OpenAI API errors**: Verify your API key is correct and has sufficient credits
3. **Web search not working**: Check your Serper API key or disable web search
4. **Vector store issues**: Delete the vector store directory and restart to rebuild

### Logs

Check application logs for detailed error information:

```bash
docker-compose logs rag-api
```

## Development

### Local Development Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables
3. Run with reload: `uvicorn main:app --reload`

### Testing

The application includes health checks and error handling. Test the API endpoints using the Swagger UI at `/docs`.

## Contributing

1. Ensure code follows Python best practices
2. Add appropriate error handling
3. Update documentation for new features
4. Test thoroughly before submitting

## License

This project is developed for research and educational purposes. 