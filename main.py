from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import logging
from contextlib import asynccontextmanager

# Import our PDF ingestion service
from ingest_pdf import init_pdf_service, pdf_service

# Original imports for the agent system
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
import re
import uuid
from datetime import datetime

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class QuestionResponse(BaseModel):
    response: str
    type: str
    session_id: str
    search_strategy: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None

class SessionResponse(BaseModel):
    session_id: str
    message: str

class ErrorResponse(BaseModel):
    error: str
    session_id: Optional[str] = None

class IngestionResponse(BaseModel):
    status: str
    message: str
    pdf_files: Optional[int] = None
    total_pages: Optional[int] = None
    total_chunks: Optional[int] = None

# Enhanced State for Multi-Agent System
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: Optional[str]
    user_query: str
    clarification_needed: bool
    search_strategy: str
    confidence_score: float
    agent_reasoning: str
    final_response: str

# Session Management
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "messages": [],
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, messages: list):
        if session_id in self.sessions:
            self.sessions[session_id]["messages"] = messages
            self.sessions[session_id]["last_activity"] = datetime.now()
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id]["messages"] = []

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Global variables for retriever and paper metadata
retriever = None
paper_metadata = {}

# Define tools
@tool
def list_papers_tool(query: str = "") -> str:
    """Lists all papers in the database with basic information like titles and page counts."""
    paper_list = []
    for paper_name, metadata in paper_metadata.items():
        first_page = metadata['first_page_content']
        lines = first_page.split('\n')
        potential_title = lines[0].strip() if lines else paper_name
        
        paper_info = f"Paper: {paper_name}\nTitle: {potential_title}\nPages: {metadata['total_pages']}\n"
        paper_list.append(paper_info)
    
    return "\n".join(paper_list)

@tool
def retriever_tool(query: str) -> str:
    """Searches and returns the most relevant information from research papers."""
    if not retriever:
        return "PDF retriever not initialized. Please check if PDFs are properly ingested."
    
    try:
        docs = retriever.invoke(query)
        
        if not docs:
            return "I found no relevant information in the documents"
        
        results = []
        seen_content = set()
        
        for i, doc in enumerate(docs):
            content_hash = hash(doc.page_content[:200])
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            paper_name = doc.metadata.get('paper_name', 'Unknown Paper')
            page_num = doc.metadata.get('page_number', 'Unknown Page')
            chunk_type = doc.metadata.get('chunk_type', 'content')
            
            result = f"**Paper: {paper_name}** (Page {page_num}, {chunk_type})\n{doc.page_content}\n"
            results.append(result)
            
            if len(results) >= 8:
                break
        
        return "\n" + "="*80 + "\n".join(results)
    except Exception as e:
        return f"Error searching documents: {str(e)}"

@tool
def search_specific_paper_tool(query: str) -> str:
    """Search within a specific paper. Format: 'paper_name: your_query'"""
    if not retriever:
        return "PDF retriever not initialized. Please check if PDFs are properly ingested."
    
    if ":" not in query:
        return "Please format your query as 'paper_name: your_question'"
    
    paper_name, search_query = query.split(":", 1)
    paper_name = paper_name.strip()
    search_query = search_query.strip()
    
    try:
        docs = retriever.invoke(search_query)
        filtered_docs = [doc for doc in docs if doc.metadata.get('paper_name', '').lower() == paper_name.lower()]
        
        if not filtered_docs:
            return f"No relevant information found in paper '{paper_name}' for query '{search_query}'"
        
        results = []
        for i, doc in enumerate(filtered_docs[:5]):
            page_num = doc.metadata.get('page_number', 'Unknown Page')
            chunk_type = doc.metadata.get('chunk_type', 'content')
            
            result = f"**{paper_name}** (Page {page_num}, {chunk_type})\n{doc.page_content}\n"
            results.append(result)
        
        return "\n" + "="*80 + "\n".join(results)
    except Exception as e:
        return f"Error searching specific paper: {str(e)}"

@tool
def web_search_tool(query: str) -> str:
    """Searches the web for current information."""
    try:
        search = GoogleSerperAPIWrapper()
        result = search.run(query)
        return f"Web Search Results:\n{result}"
    except Exception as e:
        return f"Web search failed: {str(e)}"

tools = [retriever_tool, list_papers_tool, search_specific_paper_tool, web_search_tool]
tools_dict = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# Multi-Agent System Class
class MultiAgentRAGSystem:
    def __init__(self):
        self.session_manager = SessionManager()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the multi-agent workflow graph."""
        
        def clarification_agent(state: AgentState) -> AgentState:
            """Agent to detect ambiguous queries that need clarification."""
            query = state["user_query"].lower()
            
            clarification_patterns = [
                (r'\b(enough|sufficient)\b.*\b(accuracy|performance)\b', "What specific accuracy threshold or dataset are you targeting?"),
                (r'\b(best|better)\b(?!.*\b(than|compared)\b)', "Better compared to what baseline or method?"),
                (r'\bhow many\b.*\benough\b', "Enough for what specific task or performance level?"),
                (r'\b(good|bad)\b.*\bresults?\b(?!.*\b(than|compared)\b)', "What defines 'good' results for your use case?"),
                (r'^(it|that|this|they)\b', "What specific paper, method, or concept are you referring to?"),
                (r'^(what|how|why)\s+(is|are|does|do)\s+(it|that|this)\b', "Could you specify what you're asking about?")
            ]
            
            for pattern, clarification_msg in clarification_patterns:
                if re.search(pattern, query):
                    return {
                        **state,
                        "clarification_needed": True,
                        "final_response": f"clarification: {clarification_msg}",
                        "agent_reasoning": f"Clarification agent detected pattern: {pattern}"
                    }
            
            words = query.split()
            if len(words) <= 2 and any(word in ['it', 'that', 'this', 'they'] for word in words):
                return {
                    **state,
                    "clarification_needed": True,
                    "final_response": "clarification: Could you provide more context about what you're asking?",
                    "agent_reasoning": "Clarification agent detected very short context-free query"
                }
            
            return {
                **state,
                "clarification_needed": False,
                "agent_reasoning": "Clarification agent approved query - contains searchable content"
            }
        
        def router_agent(state: AgentState) -> AgentState:
            """Agent to decide the search strategy: PDF, Web, or Both."""
            query = state["user_query"].lower()
            
            pdf_keywords = [
                "papers", "documents", "list papers", "what papers", "available papers",
                "authors", "methodology", "accuracy", "results", "dataset", "experiment",
                "citation", "abstract", "conclusion", "spider", "text-to-sql"
            ]
            
            web_keywords = [
                "recent", "latest", "this month", "this year", "current", "now",
                "openai release", "google release", "new model", "breaking news"
            ]
            
            if any(keyword in query for keyword in ["what papers", "list papers", "papers do you have", "available papers"]):
                decision = "PDF"
            elif any(keyword in query for keyword in web_keywords):
                decision = "WEB"
            elif any(keyword in query for keyword in pdf_keywords):
                decision = "PDF"
            else:
                router_prompt = f"""
                Analyze this query and decide the best search strategy: "{query}"
                
                Strategy Guidelines:
                1. **PDF Strategy** - Use for research papers, academic content, methodologies
                2. **WEB Strategy** - Use for recent events, current news, company releases
                3. **BOTH Strategy** - Use when needing both academic context and current info
                
                Respond with exactly one word: PDF, WEB, or BOTH
                """
                
                messages = [HumanMessage(content=router_prompt)]
                response = llm.invoke(messages)
                decision = response.content.strip().upper()
                
                if decision not in ["PDF", "WEB", "BOTH"]:
                    decision = "PDF"
            
            return {
                **state,
                "search_strategy": decision.lower(),
                "agent_reasoning": f"Router agent decided: {decision}"
            }
        
        def pdf_research_agent(state: AgentState) -> AgentState:
            """Agent specialized in searching and analyzing PDF content."""
            query = state["user_query"].lower()
            
            pdf_system_prompt = """
            You are a specialized research assistant focused on analyzing academic papers.
            
            Your capabilities:
            1. **retriever_tool**: Search across all papers for relevant content
            2. **search_specific_paper_tool**: Search within a specific paper (use format "paper_name: query")
            3. **list_papers_tool**: List all available papers and their metadata
            
            Guidelines:
            - Use list_papers_tool for queries about "what papers", "list papers", "available papers"
            - Use search_specific_paper_tool when the query mentions specific authors or paper names
            - For content questions, use retriever_tool to search across all papers
            - Always cite specific papers, page numbers, and section types
            - Provide exact quotes and numbers when available
            
            Focus on accuracy and scholarly precision.
            """
            
            if any(phrase in query for phrase in ["what papers", "list papers", "papers do you have", "available papers"]):
                tool_suggestion = "Use list_papers_tool to show all available papers."
            elif any(author in query for author in ["zhang", "smith", "author"]):
                tool_suggestion = "Use retriever_tool to search for author mentions and specific content."
            else:
                tool_suggestion = "Use retriever_tool to search for relevant content across all papers."
            
            messages = [
                SystemMessage(content=pdf_system_prompt),
                HumanMessage(content=f"Research this query using the PDF documents: {state['user_query']}\n\nSuggestion: {tool_suggestion}")
            ]
            
            response = llm_with_tools.invoke(messages)
            
            return {
                **state,
                "messages": state.get("messages", []) + [response],
                "agent_reasoning": state.get("agent_reasoning", "") + " | PDF research agent activated"
            }
        
        def web_research_agent(state: AgentState) -> AgentState:
            """Agent specialized in web search for current information."""
            query = state["user_query"]
            
            web_system_prompt = """
            You are a web research specialist focused on finding current, real-time information.
            Use the web_search_tool to find recent developments, news, company releases, and current data.
            Provide source URLs when possible and publication dates for currency.
            """
            
            messages = [
                SystemMessage(content=web_system_prompt),
                HumanMessage(content=f"Search the web for current information about: {query}")
            ]
            
            response = llm_with_tools.invoke(messages)
            
            return {
                **state,
                "messages": state.get("messages", []) + [response],
                "agent_reasoning": state.get("agent_reasoning", "") + " | Web research agent activated"
            }
        
        def tool_execution_agent(state: AgentState) -> AgentState:
            """Execute tool calls from research agents."""
            last_message = state["messages"][-1]
            
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return state
            
            tool_messages = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.info(f"Executing Tool: {tool_name} with args: {tool_args}")
                
                if tool_name in tools_dict:
                    try:
                        result = tools_dict[tool_name].invoke(tool_args)
                        tool_messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                name=tool_name,
                                content=str(result)
                            )
                        )
                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {str(e)}"
                        tool_messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                name=tool_name,
                                content=error_msg
                            )
                        )
                        logger.error(error_msg)
                else:
                    error_msg = f"Tool {tool_name} not found. Available: {list(tools_dict.keys())}"
                    tool_messages.append(
                        ToolMessage(
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                            content=error_msg
                        )
                    )
            
            return {
                **state,
                "messages": state["messages"] + tool_messages
            }
        
        def response_synthesis_agent(state: AgentState) -> AgentState:
            """Generate final comprehensive response."""
            query = state["user_query"]
            messages = state.get("messages", [])
            strategy = state.get("search_strategy", "pdf")
            
            tool_results = []
            for msg in messages:
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    if hasattr(msg, 'name') and msg.name in ['retriever_tool', 'list_papers_tool', 'search_specific_paper_tool', 'web_search_tool']:
                        tool_results.append(f"From {msg.name}: {msg.content}")
            
            synthesis_prompt = f"""
            Based on the research conducted, provide a comprehensive answer to: "{query}"
            
            Research results:
            {chr(10).join(tool_results) if tool_results else "No tool results available."}
            
            Your response should:
            1. Directly answer the user's question
            2. Cite specific sources with page numbers (for PDF content)
            3. Include exact quotes and numerical data when available
            4. Mention the search strategy used: {strategy}
            5. Indicate confidence level if uncertain
            6. Suggest follow-up questions if relevant
            
            Be scholarly, precise, and helpful.
            """
            
            synthesis_messages = [
                SystemMessage(content=synthesis_prompt),
                HumanMessage(content=f"Synthesize a final answer for: {query}")
            ]
            
            response = llm.invoke(synthesis_messages)
            
            return {
                **state,
                "final_response": response.content,
                "agent_reasoning": state.get("agent_reasoning", "") + " | Response synthesis completed"
            }
        
        # Router Functions
        def route_after_clarification(state: AgentState) -> str:
            return "end" if state.get("clarification_needed", False) else "router"
        
        def route_after_router(state: AgentState) -> str:
            strategy = state.get("search_strategy", "pdf").lower()
            if strategy == "both":
                return "pdf_research"
            elif strategy == "web":
                return "web_research"
            else:
                return "pdf_research"
        
        def route_after_pdf_research(state: AgentState) -> str:
            last_message = state["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tool_execution"
            elif state.get("search_strategy") == "both":
                return "web_research"
            else:
                return "response_synthesis"
        
        def route_after_web_research(state: AgentState) -> str:
            last_message = state["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tool_execution"
            else:
                return "response_synthesis"
        
        def route_after_tool_execution(state: AgentState) -> str:
            if state.get("search_strategy") == "both":
                reasoning = state.get("agent_reasoning", "")
                if "PDF research agent activated" in reasoning and "Web research agent activated" not in reasoning:
                    return "web_research"
            return "response_synthesis"
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        workflow.add_node("clarification", clarification_agent)
        workflow.add_node("router", router_agent)
        workflow.add_node("pdf_research", pdf_research_agent)
        workflow.add_node("web_research", web_research_agent)
        workflow.add_node("tool_execution", tool_execution_agent)
        workflow.add_node("response_synthesis", response_synthesis_agent)
        
        workflow.set_entry_point("clarification")
        
        workflow.add_conditional_edges(
            "clarification",
            route_after_clarification,
            {"end": END, "router": "router"}
        )
        
        workflow.add_conditional_edges(
            "router",
            route_after_router,
            {"pdf_research": "pdf_research", "web_research": "web_research"}
        )
        
        workflow.add_conditional_edges(
            "pdf_research",
            route_after_pdf_research,
            {"tool_execution": "tool_execution", "web_research": "web_research", "response_synthesis": "response_synthesis"}
        )
        
        workflow.add_conditional_edges(
            "web_research",
            route_after_web_research,
            {"tool_execution": "tool_execution", "response_synthesis": "response_synthesis"}
        )
        
        workflow.add_conditional_edges(
            "tool_execution",
            route_after_tool_execution,
            {"web_research": "web_research", "response_synthesis": "response_synthesis"}
        )
        
        workflow.add_edge("response_synthesis", END)
        
        return workflow.compile()
    
    def create_session(self) -> str:
        """Create a new session for conversation management."""
        return self.session_manager.create_session()
    
    def ask_question(self, question: str, session_id: Optional[str] = None) -> dict:
        """Main method to process questions through the multi-agent system."""
        if session_id is None:
            session_id = self.create_session()
        
        session = self.session_manager.get_session(session_id)
        if not session:
            return {"error": "Invalid session ID", "session_id": session_id}
        
        initial_state = {
            "messages": [],
            "session_id": session_id,
            "user_query": question,
            "clarification_needed": False,
            "search_strategy": "",
            "confidence_score": 0.0,
            "agent_reasoning": "",
            "final_response": ""
        }
        
        logger.info(f"Processing query: {question} (Session: {session_id})")
        
        try:
            result = self.graph.invoke(initial_state)
            
            if result.get("clarification_needed", False):
                return {
                    "response": result["final_response"],
                    "type": "clarification",
                    "session_id": session_id,
                    "reasoning": result.get("agent_reasoning", "")
                }
            
            session_messages = session.get("messages", [])
            session_messages.extend([question, result["final_response"]])
            self.session_manager.update_session(session_id, session_messages)
            
            return {
                "response": result["final_response"],
                "type": "answer",
                "session_id": session_id,
                "search_strategy": result.get("search_strategy", "unknown"),
                "reasoning": result.get("agent_reasoning", ""),
                "confidence": result.get("confidence_score", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "error": f"Error processing query: {str(e)}",
                "session_id": session_id,
                "type": "error"
            }
    
    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        self.session_manager.clear_session(session_id)
        return True
    
    def get_session_history(self, session_id: str) -> list:
        """Get conversation history for a session."""
        session = self.session_manager.get_session(session_id)
        return session.get("messages", []) if session else []

# Global instance
multi_agent_rag = None

async def startup_event():
    """Initialize services on startup."""
    global retriever, paper_metadata, multi_agent_rag
    
    try:
        logger.info("Initializing PDF service...")
        pdf_svc = init_pdf_service()
        retriever = pdf_svc.get_retriever()
        paper_metadata = pdf_svc.paper_metadata
        
        logger.info("Initializing multi-agent RAG system...")
        multi_agent_rag = MultiAgentRAGSystem()
        
        logger.info("Startup complete!")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent RAG System",
    description="A sophisticated multi-agent system for research paper analysis and web search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multi-Agent RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "POST /ask - Ask a question",
            "session": "POST /session - Create new session",
            "session_history": "GET /session/{session_id}/history - Get session history",
            "clear_session": "DELETE /session/{session_id} - Clear session",
            "ingest": "POST /ingest - Re-ingest PDFs",
            "health": "GET /health - Health check"
        }
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the multi-agent system."""
    if not multi_agent_rag:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = multi_agent_rag.ask_question(request.question, request.session_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return QuestionResponse(**result)
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/session", response_model=SessionResponse)
async def create_session():
    """Create a new conversation session."""
    if not multi_agent_rag:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        session_id = multi_agent_rag.create_session()
        return SessionResponse(session_id=session_id, message="Session created successfully")
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    if not multi_agent_rag:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        history = multi_agent_rag.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if not multi_agent_rag:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        success = multi_agent_rag.clear_session(session_id)
        if success:
            return {"message": "Session cleared successfully", "session_id": session_id}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ingest", response_model=IngestionResponse)
async def reingest_pdfs(background_tasks: BackgroundTasks):
    """Re-ingest PDFs (useful for when new PDFs are added)."""
    try:
        result = pdf_service.ingest_pdfs()
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        # Update global variables
        global retriever, paper_metadata
        retriever = pdf_service.get_retriever()
        paper_metadata = pdf_service.paper_metadata
        
        return IngestionResponse(**result)
    except Exception as e:
        logger.error(f"Error re-ingesting PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global retriever, paper_metadata, multi_agent_rag
    
    status = {
        "status": "healthy",
        "pdf_service": retriever is not None,
        "papers_loaded": len(paper_metadata) if paper_metadata else 0,
        "agent_system": multi_agent_rag is not None
    }
    
    if not all([retriever, multi_agent_rag]):
        status["status"] = "degraded"
        return status, 503
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)