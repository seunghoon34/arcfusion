from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
import re
import uuid
from datetime import datetime

load_dotenv()

# Enhanced State for Multi-Agent System
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: Optional[str]
    user_query: str
    clarification_needed: bool
    search_strategy: str  # "pdf", "web", "both", "clarify"
    confidence_score: float
    agent_reasoning: str
    final_response: str

# Session Management for RESTful API support
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

# Load and process PDFs (keeping your existing logic)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

papers_folder = "papers"

if not os.path.exists(papers_folder):
    raise FileNotFoundError(f"Papers folder not found: {papers_folder}")

# Get all PDF files from the papers folder
pdf_files = []
for file in os.listdir(papers_folder):
    if file.lower().endswith('.pdf'):
        pdf_files.append(os.path.join(papers_folder, file))

if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in folder: {papers_folder}")

print(f"Found {len(pdf_files)} PDF files to process")

# Load all PDFs and preserve metadata
all_pages = []
paper_metadata = {}  # Store paper-level metadata

for pdf_file in pdf_files:
    try:
        pdf_loader = PyPDFLoader(pdf_file)
        pages = pdf_loader.load()
        
        # Extract paper filename for reference
        paper_name = os.path.basename(pdf_file).replace('.pdf', '')
        
        # Add paper source to each page's metadata
        for i, page in enumerate(pages):
            page.metadata['paper_name'] = paper_name
            page.metadata['paper_path'] = pdf_file
            page.metadata['page_number'] = i + 1
            
            # Try to extract title and authors from first few pages
            if i < 3:  # Check first 3 pages for title/authors
                text = page.page_content.lower()
                if 'abstract' in text or 'introduction' in text:
                    # This is likely a research paper format
                    page.metadata['likely_metadata_page'] = True
        
        all_pages.extend(pages)
        paper_metadata[paper_name] = {
            'path': pdf_file,
            'total_pages': len(pages),
            'first_page_content': pages[0].page_content[:500] if pages else ""
        }
        
        print(f"Loaded {pdf_file}: {len(pages)} pages")
    except Exception as e:
        print(f"Error loading {pdf_file}: {e}")
        continue

print(f"Total pages loaded from all PDFs: {len(all_pages)}")

# Improved chunking strategy for research papers
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # Increased for better context
    chunk_overlap=400,  # Increased overlap
    separators=["\n\n", "\n", ". ", " ", ""]  # Better separators for academic text
)

pages_split = text_splitter.split_documents(all_pages)

# Enhance chunk metadata
for chunk in pages_split:
    # Add more context to each chunk
    chunk.metadata['chunk_type'] = 'content'
    text_lower = chunk.page_content.lower()
    
    # Identify chunk types
    if any(keyword in text_lower for keyword in ['abstract', 'summary']):
        chunk.metadata['chunk_type'] = 'abstract'
    elif any(keyword in text_lower for keyword in ['introduction', 'background']):
        chunk.metadata['chunk_type'] = 'introduction'
    elif any(keyword in text_lower for keyword in ['conclusion', 'discussion']):
        chunk.metadata['chunk_type'] = 'conclusion'
    elif any(keyword in text_lower for keyword in ['method', 'approach', 'algorithm']):
        chunk.metadata['chunk_type'] = 'methodology'
    elif any(keyword in text_lower for keyword in ['result', 'experiment', 'evaluation']):
        chunk.metadata['chunk_type'] = 'results'
    elif any(keyword in text_lower for keyword in ['reference', 'bibliography']):
        chunk.metadata['chunk_type'] = 'references'

persist_directory = r"/Users/seunghoonhan/langgraph"
collection_name = "papers_collection_improved"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store with {len(pages_split)} chunks!")
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# Improved retriever with multiple search strategies
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 15}  # Increased from 5
)

# Define tools (keeping your existing tools)
@tool
def list_papers_tool(query: str = "") -> str:
    """
    Lists all papers in the database with basic information like titles and page counts.
    """
    paper_list = []
    for paper_name, metadata in paper_metadata.items():
        # Try to extract title from first page
        first_page = metadata['first_page_content']
        lines = first_page.split('\n')
        potential_title = lines[0].strip() if lines else paper_name
        
        paper_info = f"Paper: {paper_name}\nTitle: {potential_title}\nPages: {metadata['total_pages']}\n"
        paper_list.append(paper_info)
    
    return "\n".join(paper_list)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the most relevant information from the research papers.
    It includes paper names, page numbers, and section types for better context.
    """
    docs = retriever.invoke(query)
    
    if not docs:
        return "I found no relevant information in the documents"
    
    results = []
    seen_content = set()  # Avoid duplicate content
    
    for i, doc in enumerate(docs):
        # Skip if we've seen very similar content
        content_hash = hash(doc.page_content[:200])
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        
        paper_name = doc.metadata.get('paper_name', 'Unknown Paper')
        page_num = doc.metadata.get('page_number', 'Unknown Page')
        chunk_type = doc.metadata.get('chunk_type', 'content')
        
        result = f"**Paper: {paper_name}** (Page {page_num}, {chunk_type})\n{doc.page_content}\n"
        results.append(result)
        
        # Limit to top 8 most relevant chunks to avoid overwhelming
        if len(results) >= 8:
            break
    
    return "\n" + "="*80 + "\n".join(results)

@tool
def search_specific_paper_tool(query: str) -> str:
    """
    Search within a specific paper. Format: "paper_name: your_query"
    Example: "paper1: methodology for classification"
    """
    if ":" not in query:
        return "Please format your query as 'paper_name: your_question'"
    
    paper_name, search_query = query.split(":", 1)
    paper_name = paper_name.strip()
    search_query = search_query.strip()
    
    # Search with paper name filter
    docs = retriever.invoke(search_query)
    
    # Filter results to specific paper
    filtered_docs = [doc for doc in docs if doc.metadata.get('paper_name', '').lower() == paper_name.lower()]
    
    if not filtered_docs:
        return f"No relevant information found in paper '{paper_name}' for query '{search_query}'"
    
    results = []
    for i, doc in enumerate(filtered_docs[:5]):  # Top 5 results
        page_num = doc.metadata.get('page_number', 'Unknown Page')
        chunk_type = doc.metadata.get('chunk_type', 'content')
        
        result = f"**{paper_name}** (Page {page_num}, {chunk_type})\n{doc.page_content}\n"
        results.append(result)
    
    return "\n" + "="*80 + "\n".join(results)

@tool
def web_search_tool(query: str) -> str:
    """
    This tool searches the web for current information.
    """
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
        
        # Agent 1: Clarification Agent
        def clarification_agent(state: AgentState) -> AgentState:
            """Agent to detect ONLY truly ambiguous queries that need clarification."""
            query = state["user_query"].lower()
            
            # Only ask for clarification for very specific problematic patterns
            clarification_patterns = [
                # Vague quantifiers without any context
                (r'\b(enough|sufficient)\b.*\b(accuracy|performance)\b', "What specific accuracy threshold or dataset are you targeting?"),
                (r'\b(best|better)\b(?!.*\b(than|compared)\b)', "Better compared to what baseline or method?"),
                (r'\bhow many\b.*\benough\b', "Enough for what specific task or performance level?"),
                (r'\b(good|bad)\b.*\bresults?\b(?!.*\b(than|compared)\b)', "What defines 'good' results for your use case?"),
                
                # Completely context-free pronouns at start
                (r'^(it|that|this|they)\b', "What specific paper, method, or concept are you referring to?"),
                
                # Questions with no searchable content
                (r'^(what|how|why)\s+(is|are|does|do)\s+(it|that|this)\b', "Could you specify what you're asking about?")
            ]
            
            # Check if query matches problematic patterns
            for pattern, clarification_msg in clarification_patterns:
                if re.search(pattern, query):
                    return {
                        **state,
                        "clarification_needed": True,
                        "final_response": f"clarification: {clarification_msg}",
                        "agent_reasoning": f"Clarification agent detected pattern: {pattern}"
                    }
            
            # Special handling for very short, context-free queries
            words = query.split()
            if len(words) <= 2 and any(word in ['it', 'that', 'this', 'they'] for word in words):
                return {
                    **state,
                    "clarification_needed": True,
                    "final_response": "clarification: Could you provide more context about what you're asking?",
                    "agent_reasoning": "Clarification agent detected very short context-free query"
                }
            
            # For searchable queries (like "zhang et al", "methodology", etc.), proceed
            # These can be handled by the retrieval system
            return {
                **state,
                "clarification_needed": False,
                "agent_reasoning": "Clarification agent approved query - contains searchable content"
            }
        
        # Agent 2: Router Agent
        def router_agent(state: AgentState) -> AgentState:
            """Agent to decide the search strategy: PDF, Web, or Both."""
            query = state["user_query"].lower()
            
            # Rule-based routing for common patterns
            pdf_keywords = [
                "papers", "documents", "list papers", "what papers", "available papers",
                "authors", "methodology", "accuracy", "results", "dataset", "experiment",
                "citation", "abstract", "conclusion", "spider", "text-to-sql"
            ]
            
            web_keywords = [
                "recent", "latest", "this month", "this year", "current", "now",
                "openai release", "google release", "new model", "breaking news"
            ]
            
            # Check for direct matches first
            if any(keyword in query for keyword in ["what papers", "list papers", "papers do you have", "available papers"]):
                decision = "PDF"
            elif any(keyword in query for keyword in web_keywords):
                decision = "WEB"
            elif any(keyword in query for keyword in pdf_keywords):
                decision = "PDF"
            else:
                # Use LLM for complex queries
                router_prompt = f"""
                Analyze this query and decide the best search strategy: "{query}"
                
                Strategy Guidelines:
                
                1. **PDF Strategy** - Use when query asks about:
                   - Specific research papers, authors, methodologies
                   - Academic results, experiments, datasets mentioned in papers
                   - Citations like "Zhang et al. (2024)", "Spider dataset"
                   - Technical details from research (accuracy scores, prompt templates)
                   - Content that would be in academic papers
                
                2. **WEB Strategy** - Use when query asks about:
                   - Recent events, current news, "this month", "recently released"
                   - Companies and their latest products/releases
                   - Real-time information, current status
                   - General knowledge not specific to research papers
                
                3. **BOTH Strategy** - Use when query needs:
                   - Comparison between paper content and current state
                   - Academic context + current developments
                   - Background from papers + recent updates
                
                Examples:
                - "Which prompt template gave highest accuracy on Spider in Zhang et al.?" → PDF
                - "What did OpenAI release this month?" → WEB
                - "How do the results in these papers compare to current SOTA?" → BOTH
                
                Respond with exactly one word: PDF, WEB, or BOTH
                """
                
                messages = [HumanMessage(content=router_prompt)]
                response = llm.invoke(messages)
                decision = response.content.strip().upper()
                
                # Ensure valid decision
                if decision not in ["PDF", "WEB", "BOTH"]:
                    decision = "PDF"  # Default to PDF for academic queries
            
            return {
                **state,
                "search_strategy": decision.lower(),
                "agent_reasoning": f"Router agent decided: {decision}"
            }
        
        # Agent 3: PDF Research Agent
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
            - Include confidence indicators if information is uncertain
            
            Focus on accuracy and scholarly precision.
            """
            
            # Determine which tool to use based on query
            if any(phrase in query for phrase in ["what papers", "list papers", "papers do you have", "available papers"]):
                tool_suggestion = "Use list_papers_tool to show all available papers."
            elif any(author in query for author in ["zhang", "smith", "author"]):
                tool_suggestion = "Use retriever_tool to search for author mentions and specific content. The retriever_tool works better for author-based queries."
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
        
        # Agent 4: Web Research Agent
        def web_research_agent(state: AgentState) -> AgentState:
            """Agent specialized in web search for current information."""
            query = state["user_query"]
            
            web_system_prompt = """
            You are a web research specialist focused on finding current, real-time information.
            
            Use the web_search_tool to find:
            - Recent developments and news
            - Current company releases and announcements
            - Latest research and publications
            - Real-time data and statistics
            
            Provide:
            - Source URLs when possible
            - Publication dates for currency
            - Clear distinction between verified facts and claims
            - Context about reliability of sources
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
        
        # Agent 5: Tool Execution Agent
        def tool_execution_agent(state: AgentState) -> AgentState:
            """Execute tool calls from research agents."""
            last_message = state["messages"][-1]
            
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                return state
            
            tool_messages = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print(f"Executing Tool: {tool_name} with args: {tool_args}")
                
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
                        print(f"Tool {tool_name} executed successfully")
                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {str(e)}"
                        tool_messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                name=tool_name,
                                content=error_msg
                            )
                        )
                        print(error_msg)
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
        
        # Agent 6: Response Synthesis Agent
        def response_synthesis_agent(state: AgentState) -> AgentState:
            """Generate final comprehensive response."""
            query = state["user_query"]
            messages = state.get("messages", [])
            strategy = state.get("search_strategy", "pdf")
            
            # Extract tool results content for synthesis (avoid tool call format issues)
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
            
            Format:
            - Start with a direct answer
            - Provide supporting evidence
            - End with any caveats or additional context
            
            Be scholarly, precise, and helpful.
            """
            
            # Use simple message format to avoid tool call issues
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
                return "pdf_research"  # Start with PDF, then web
            elif strategy == "web":
                return "web_research"
            else:
                return "pdf_research"  # Default to PDF
        
        def route_after_pdf_research(state: AgentState) -> str:
            last_message = state["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tool_execution"
            elif state.get("search_strategy") == "both":
                return "web_research"  # Continue to web for "both" strategy
            else:
                return "response_synthesis"
        
        def route_after_web_research(state: AgentState) -> str:
            last_message = state["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tool_execution"
            else:
                return "response_synthesis"
        
        def route_after_tool_execution(state: AgentState) -> str:
            # Check if we need to continue with the other search strategy
            if state.get("search_strategy") == "both":
                # Check if we've done both PDF and web research
                reasoning = state.get("agent_reasoning", "")
                if "PDF research agent activated" in reasoning and "Web research agent activated" not in reasoning:
                    return "web_research"
            return "response_synthesis"
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add all agents as nodes
        workflow.add_node("clarification", clarification_agent)
        workflow.add_node("router", router_agent)
        workflow.add_node("pdf_research", pdf_research_agent)
        workflow.add_node("web_research", web_research_agent)
        workflow.add_node("tool_execution", tool_execution_agent)
        workflow.add_node("response_synthesis", response_synthesis_agent)
        
        # Define the workflow
        workflow.set_entry_point("clarification")
        
        workflow.add_conditional_edges(
            "clarification",
            route_after_clarification,
            {
                "end": END,
                "router": "router"
            }
        )
        
        workflow.add_conditional_edges(
            "router",
            route_after_router,
            {
                "pdf_research": "pdf_research",
                "web_research": "web_research"
            }
        )
        
        workflow.add_conditional_edges(
            "pdf_research",
            route_after_pdf_research,
            {
                "tool_execution": "tool_execution",
                "web_research": "web_research",
                "response_synthesis": "response_synthesis"
            }
        )
        
        workflow.add_conditional_edges(
            "web_research",
            route_after_web_research,
            {
                "tool_execution": "tool_execution",
                "response_synthesis": "response_synthesis"
            }
        )
        
        workflow.add_conditional_edges(
            "tool_execution",
            route_after_tool_execution,
            {
                "web_research": "web_research",
                "response_synthesis": "response_synthesis"
            }
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
        
        # Prepare initial state
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
        
        print(f"\n=== PROCESSING QUERY ===")
        print(f"Question: {question}")
        print(f"Session: {session_id}")
        
        try:
            # Run the multi-agent workflow
            result = self.graph.invoke(initial_state)
            
            # Handle clarification requests
            if result.get("clarification_needed", False):
                return {
                    "response": result["final_response"],
                    "type": "clarification",
                    "session_id": session_id,
                    "reasoning": result.get("agent_reasoning", "")
                }
            
            # Update session history
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

# Initialize the multi-agent system
multi_agent_rag = MultiAgentRAGSystem()

def running_agent():
    """Console interface for testing the multi-agent system."""
    print("\n=== MULTI-AGENT RAG SYSTEM ===")
    print("This system uses 6 specialized agents:")
    print("1. Clarification Agent - Detects ambiguous queries")
    print("2. Router Agent - Decides search strategy (PDF/Web/Both)")
    print("3. PDF Research Agent - Searches academic papers")
    print("4. Web Research Agent - Searches current information")
    print("5. Tool Execution Agent - Executes research tools")
    print("6. Response Synthesis Agent - Creates final answers")
    print("\nCommands:")
    print("- 'new session' - Start a new conversation session")
    print("- 'clear' - Clear current session")
    print("- 'exit' or 'quit' - Exit the system")
    print("- Ask any question about research papers or current topics")
    
    session_id = multi_agent_rag.create_session()
    print(f"\nSession created: {session_id}")
    
    while True:
        user_input = input("\nYour question: ")
        
        if user_input.lower() in ['exit', 'quit']:
            break
        
        if user_input.lower() == 'new session':
            session_id = multi_agent_rag.create_session()
            print(f"New session created: {session_id}")
            continue
            
        if user_input.lower() == 'clear':
            multi_agent_rag.clear_session(session_id)
            print("Session history cleared")
            continue
        
        # Process the question
        result = multi_agent_rag.ask_question(user_input, session_id)
        
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
        else:
            print(f"\n=== RESPONSE ===")
            print(f"Type: {result.get('type', 'unknown')}")
            if result.get('search_strategy'):
                print(f"Strategy: {result['search_strategy'].upper()}")
            print(f"\n{result['response']}")
            
            if result.get('reasoning'):
                print(f"\n🔍 Agent Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    running_agent()