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


## Development

### Local Development Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables
3. Run with reload: `uvicorn main:app --reload`

### Testing

The application includes health checks and error handling. Test the API endpoints using the Swagger UI at `/docs`.

## Architecture Trade-offs & Future Improvements

### Current Trade-offs

This system was designed with several conscious trade-offs for simplicity and rapid development:

#### **1. Session Storage**
- **Current**: In-memory session storage using Python dictionaries
- **Trade-off**: Simple but not persistent across restarts, limited to single instance
- **Limitation**: Sessions lost on application restart, no horizontal scaling

#### **2. Confidence Scoring**
- **Current**: Rule-based confidence calculation using predefined factors
- **Trade-off**: Simple and interpretable but not adaptive or learning-based
- **Limitation**: May not accurately reflect actual answer quality over time

#### **3. Agent Routing**
- **Current**: Fixed routing logic with predefined patterns and fallback to LLM
- **Trade-off**: Predictable but not optimized for individual query types
- **Limitation**: Cannot learn from routing successes/failures

#### **4. Document Processing**
- **Current**: Basic PDF text extraction with simple chunking
- **Trade-off**: Fast processing but loses document structure and formatting
- **Limitation**: Cannot process tables, images, or complex layouts effectively

#### **5. Vector Storage**
- **Current**: Local ChromaDB instance with basic embedding strategy
- **Trade-off**: Simple setup but limited scalability and search sophistication
- **Limitation**: No semantic clustering, limited to text similarity

#### **6. Tool Architecture**
- **Current**: Fixed set of predefined tools (PDF search, web search, paper listing)
- **Trade-off**: Reliable but not extensible without code changes
- **Limitation**: Cannot dynamically add new data sources or capabilities

#### **7. Processing Model**
- **Current**: Synchronous processing with sequential agent execution
- **Trade-off**: Simple debugging but slower response times
- **Limitation**: Cannot handle multiple queries efficiently or leverage parallelism

### Future Improvement Opportunities

#### **Short-term Improvements**

1. **Persistent Session Storage**
   - Implement Redis or PostgreSQL for session persistence
   - Add session expiration and cleanup mechanisms
   - Enable horizontal scaling across multiple instances

2. **Enhanced Conversation Memory**
   - Implement conversation summarization for long sessions
   - Add memory compression to handle context length limits
   - Store user preferences and interaction patterns

3. **Better Error Handling**
   - Add comprehensive retry mechanisms for API failures
   - Implement graceful degradation when tools fail
   - Add detailed error logging and monitoring

4. **Performance Optimizations**
   - Implement async processing for tool execution
   - Add response caching for common queries
   - Optimize vector search with better indexing

5. **Enhanced Document Indexing**
   - Add metadata indexes for fast paper name and author lookups
   - Implement chunk type indexing (abstract, methodology, results, conclusion)
   - Create page range indexes for targeted section searches
   - Add composite indexes for multi-field queries (paper + section type)
   - Implement full-text search indexes for exact phrase matching
   - Add temporal indexing for publication dates and citation tracking

6. **Basic Security & Scaling**
   - Add API rate limiting and request throttling
   - Implement basic input validation and sanitization
   - Add health checks and readiness probes for containers
   - Implement horizontal scaling with load balancing
   - Add basic authentication mechanisms (API keys)

#### **Medium-term Improvements**

1. **Advanced Document Processing**
   - Support for tables, images, and structured data extraction
   - Multi-modal document understanding (OCR, layout analysis)
   - Document relationship mapping and citation tracking

2. **Intelligent Agent Routing**
   - Machine learning-based routing decisions
   - Performance tracking and adaptive routing
   - User feedback integration for routing optimization

3. **Enhanced Confidence Scoring**
   - ML-based confidence prediction using query-response pairs
   - Multi-factor confidence including source reliability
   - Confidence calibration and continuous improvement

4. **Advanced Search Capabilities**
   - Semantic clustering and topic modeling
   - Multi-hop reasoning across documents
   - Temporal reasoning and trend analysis

5. **Enhanced Security & Scaling**
   - Implement OAuth 2.0 and JWT-based authentication
   - Add encryption for data at rest and in transit
   - Implement comprehensive input validation and prompt injection protection
   - Add microservices architecture for better scaling
   - Implement auto-scaling based on load metrics
   - Add comprehensive security audit logging

#### **Long-term Improvements**

1. **Plugin Architecture**
   - Extensible tool system for custom data sources
   - Dynamic tool discovery and registration
   - Third-party plugin ecosystem

2. **Multi-modal Capabilities**
   - Support for image, audio, and video content
   - Cross-modal search and reasoning
   - Visual document understanding

3. **Advanced AI Features**
   - Multi-agent collaboration and consensus
   - Reasoning chain visualization and explanation
   - Automated fact-checking and source verification

4. **Enterprise Features**
   - Role-based access control and user management
   - Audit logging and compliance features
   - Integration with enterprise systems (SSO, etc.)

5. **Evaluation & Monitoring**
   - Automated answer quality assessment
   - A/B testing framework for system improvements
   - Real-time performance monitoring and alerting


