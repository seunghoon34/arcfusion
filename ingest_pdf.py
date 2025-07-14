import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFIngestionService:
    """Service for ingesting and processing PDF documents."""
    
    def __init__(self, papers_folder: str = "papers", persist_directory: str = "./vectorstore"):
        self.papers_folder = papers_folder
        self.persist_directory = persist_directory
        self.collection_name = "papers_collection_improved"
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
        self.paper_metadata = {}
        
    def validate_papers_folder(self) -> List[str]:
        """Validate papers folder exists and get PDF files."""
        if not os.path.exists(self.papers_folder):
            raise FileNotFoundError(f"Papers folder not found: {self.papers_folder}")
        
        # Get all PDF files from the papers folder
        pdf_files = []
        for file in os.listdir(self.papers_folder):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.papers_folder, file))
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in folder: {self.papers_folder}")
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        return pdf_files
    
    def load_pdfs(self, pdf_files: List[str]) -> List[Any]:
        """Load all PDFs and preserve metadata."""
        all_pages = []
        
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
                self.paper_metadata[paper_name] = {
                    'path': pdf_file,
                    'total_pages': len(pages),
                    'first_page_content': pages[0].page_content[:500] if pages else ""
                }
                
                logger.info(f"Loaded {pdf_file}: {len(pages)} pages")
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
                continue
        
        logger.info(f"Total pages loaded from all PDFs: {len(all_pages)}")
        return all_pages
    
    def split_documents(self, pages: List[Any]) -> List[Any]:
        """Improved chunking strategy for research papers."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Increased for better context
            chunk_overlap=400,  # Increased overlap
            separators=["\n\n", "\n", ". ", " ", ""]  # Better separators for academic text
        )
        
        pages_split = text_splitter.split_documents(pages)
        
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
        
        return pages_split
    
    def create_vectorstore(self, documents: List[Any]) -> Chroma:
        """Create and populate the vector store."""
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            logger.info(f"Created ChromaDB vector store with {len(documents)} chunks!")
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error setting up ChromaDB: {str(e)}")
            raise
    
    def get_retriever(self, k: int = 15):
        """Get the retriever for the vectorstore."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call ingest_pdfs() first.")
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def ingest_pdfs(self) -> Dict[str, Any]:
        """Main method to ingest all PDFs and create vectorstore."""
        try:
            # Validate and get PDF files
            pdf_files = self.validate_papers_folder()
            
            # Load all PDFs
            all_pages = self.load_pdfs(pdf_files)
            
            # Split documents into chunks
            documents = self.split_documents(all_pages)
            
            # Create vectorstore
            vectorstore = self.create_vectorstore(documents)
            
            return {
                "status": "success",
                "message": f"Successfully ingested {len(pdf_files)} PDF files",
                "pdf_files": len(pdf_files),
                "total_pages": len(all_pages),
                "total_chunks": len(documents),
                "paper_metadata": self.paper_metadata
            }
            
        except Exception as e:
            logger.error(f"Error during PDF ingestion: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during PDF ingestion: {str(e)}"
            }
    
    def check_vectorstore_exists(self) -> bool:
        """Check if vectorstore already exists."""
        collection_path = os.path.join(self.persist_directory, self.collection_name)
        return os.path.exists(collection_path)
    
    def load_existing_vectorstore(self) -> Chroma:
        """Load existing vectorstore if it exists."""
        if not self.check_vectorstore_exists():
            raise ValueError("No existing vectorstore found. Run ingest_pdfs() first.")
        
        try:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            logger.info("Loaded existing vectorstore")
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error loading existing vectorstore: {str(e)}")
            raise

# Initialize the PDF ingestion service
pdf_service = PDFIngestionService()

def init_pdf_service():
    """Initialize the PDF service and ingest PDFs if needed."""
    if not pdf_service.check_vectorstore_exists():
        logger.info("No existing vectorstore found. Ingesting PDFs...")
        result = pdf_service.ingest_pdfs()
        if result["status"] == "error":
            raise RuntimeError(result["message"])
    else:
        logger.info("Loading existing vectorstore...")
        pdf_service.load_existing_vectorstore()
    
    return pdf_service
