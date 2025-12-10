"""
Baseline RAG System for Financial Document Question Answering.

This module provides a baseline Retrieval-Augmented Generation (RAG) system
that uses LangChain's RetrievalQA chain to answer questions about financial
documents with proper source citations.

Features:
    - Uses GPT-4 Turbo for high-quality answers
    - Enforces citation format [1], [2], etc.
    - Tracks latency and token usage for cost monitoring
    - Integrates with any BaseVectorStore implementation

Usage:
    from src.vector_store import get_vector_store
    from src.baseline_rag import BaselineRAG
    
    vector_store = get_vector_store(mode="chroma")
    rag = BaselineRAG(vector_store)
    result = rag.query("What was the company's revenue in Q4 2023?")
    print(result["answer"])
"""

import logging
import time
from typing import Any, Dict, List, Optional

import tiktoken
from langchain_classic.chains import RetrievalQA
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from src.vector_store import BaseVectorStore

# Configure logging
logger = logging.getLogger(__name__)


# Prompt template that enforces citations
RAG_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question. 
If you don't know the answer, say 'I don't have sufficient information.'
Include citations using [1], [2] format for each source used.

Context: {context}
Question: {question}

Answer with citations:"""


class TokenCountingCallback(BaseCallbackHandler):
    """
    Callback handler to count tokens during LLM calls.
    
    This handler tracks both prompt and completion tokens for accurate
    cost estimation and usage monitoring.
    
    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used (prompt + completion)
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the token counting callback.
        
        Args:
            model_name: Name of the model for accurate token counting
        """
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.model_name = model_name
        
        # Initialize tiktoken encoder for the model
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for GPT-4 models
            self._encoding = tiktoken.get_encoding("cl100k_base")
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Count tokens when LLM starts processing."""
        for prompt in prompts:
            self.prompt_tokens += len(self._encoding.encode(prompt))
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Count tokens when LLM finishes processing."""
        if hasattr(response, 'generations'):
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, 'text'):
                        self.completion_tokens += len(
                            self._encoding.encode(generation.text)
                        )
        
        self.total_tokens = self.prompt_tokens + self.completion_tokens
    
    def reset(self) -> None:
        """Reset all token counts."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0


class VectorStoreRetriever(BaseRetriever):
    """
    Custom retriever that wraps BaseVectorStore for LangChain compatibility.
    
    This adapter allows any BaseVectorStore implementation to be used
    with LangChain's RetrievalQA chain.
    
    Attributes:
        vector_store: The underlying vector store implementation
        k: Number of documents to retrieve
    """
    
    vector_store: Any  # BaseVectorStore type
    k: int = 5
    _retrieved_docs_with_scores: List = PrivateAttr(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query string
            
        Returns:
            List of relevant Document objects
        """
        # Perform similarity search
        results = self.vector_store.similarity_search(query, k=self.k)
        
        # Store results with scores for later reference
        self._retrieved_docs_with_scores = results
        
        # Extract just the documents for the chain
        documents = []
        for i, (doc, score) in enumerate(results, start=1):
            # Add source index to metadata for citation tracking
            doc.metadata["source_index"] = i
            doc.metadata["similarity_score"] = score
            documents.append(doc)
        
        logger.debug(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
        return documents
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of document retrieval."""
        # For now, use sync version as vector stores may not support async
        return self._get_relevant_documents(query)
    
    def get_retrieved_docs_with_scores(self) -> List:
        """Return the last retrieved documents with their scores."""
        return self._retrieved_docs_with_scores


class BaselineRAG:
    """
    Baseline RAG system for financial document question answering.
    
    This class implements a simple but effective RAG pipeline using:
    - GPT-4 Turbo for answer generation
    - Vector similarity search for document retrieval
    - Citation enforcement in prompts
    - Token counting for cost tracking
    
    The system retrieves relevant document chunks from the vector store,
    constructs a context-aware prompt, and generates an answer with
    proper source citations.
    
    Attributes:
        vector_store: The vector store containing document embeddings
        llm: The ChatOpenAI language model
        retriever: Custom retriever wrapping the vector store
        chain: The RetrievalQA chain
        token_callback: Callback for tracking token usage
    
    Example:
        >>> from src.vector_store import get_vector_store
        >>> from src.baseline_rag import BaselineRAG
        >>> 
        >>> store = get_vector_store(mode="chroma")
        >>> rag = BaselineRAG(store)
        >>> result = rag.query("What was the total revenue?")
        >>> print(result["answer"])
        >>> print(f"Latency: {result['latency']:.2f}s")
        >>> print(f"Tokens used: {result['token_count']}")
    """
    
    # Model configuration
    MODEL_NAME = "gpt-4o"  # Updated from gpt-4-turbo-preview for better performance and cost
    TEMPERATURE = 0.2
    DEFAULT_K = 5
    
    def __init__(
        self, 
        vector_store: BaseVectorStore,
        k: int = DEFAULT_K,
        temperature: float = TEMPERATURE
    ):
        """
        Initialize the Baseline RAG system.
        
        Args:
            vector_store: A BaseVectorStore instance containing document embeddings.
                         Can be ChromaVectorStore for local development or
                         PineconeVectorStore for production.
            k: Number of documents to retrieve for context (default: 5)
            temperature: LLM temperature for response generation (default: 0.2)
        
        Raises:
            ValueError: If vector_store is None
        """
        if vector_store is None:
            raise ValueError("vector_store cannot be None")
        
        self.vector_store = vector_store
        self.k = k
        
        # Initialize token counting callback
        self.token_callback = TokenCountingCallback(model_name=self.MODEL_NAME)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.MODEL_NAME,
            temperature=temperature,
            callbacks=[self.token_callback]
        )
        
        # Create custom retriever
        self.retriever = VectorStoreRetriever(
            vector_store=vector_store,
            k=k
        )
        
        # Create prompt template
        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Build the RetrievalQA chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        logger.info(
            f"BaselineRAG initialized: model={self.MODEL_NAME}, "
            f"temperature={temperature}, k={k}"
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        This method performs the full RAG pipeline:
        1. Retrieves relevant documents from the vector store
        2. Constructs a prompt with the retrieved context
        3. Generates an answer using GPT-4 Turbo
        4. Tracks latency and token usage
        
        Args:
            question: The question to answer about the financial documents
        
        Returns:
            A dictionary containing:
                - answer (str): The generated answer with citations
                - sources (List[Document]): The source documents used
                - latency (float): Query processing time in seconds
                - token_count (int): Total tokens used (prompt + completion)
        
        Raises:
            RuntimeError: If the query fails due to LLM or retrieval errors
        
        Example:
            >>> result = rag.query("What was the revenue growth in 2023?")
            >>> print(result["answer"])
            "According to the financial report, revenue grew by 15% in 2023 [1].
             This growth was primarily driven by increased cloud services
             adoption [2]."
            >>> print(f"Used {len(result['sources'])} sources")
            "Used 5 sources"
        """
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": [],
                "latency": 0.0,
                "token_count": 0
            }
        
        # Reset token counter for this query
        self.token_callback.reset()
        
        # Track start time
        start_time = time.perf_counter()
        
        try:
            # Execute the chain
            result = self.chain.invoke({"query": question})
            
            # Calculate latency
            latency = time.perf_counter() - start_time
            
            # Extract answer and sources
            answer = result.get("result", "")
            source_documents = result.get("source_documents", [])
            
            # Get token count from callback
            token_count = self.token_callback.total_tokens
            
            logger.info(
                f"Query completed: latency={latency:.2f}s, "
                f"tokens={token_count}, sources={len(source_documents)}"
            )
            
            return {
                "answer": answer,
                "sources": source_documents,
                "latency": latency,
                "token_count": token_count
            }
            
        except Exception as e:
            latency = time.perf_counter() - start_time
            logger.error(f"Query failed after {latency:.2f}s: {e}")
            raise RuntimeError(f"RAG query failed: {e}")
    
    def get_cost_estimate(self, token_count: int) -> Dict[str, float]:
        """
        Estimate the cost for a given token count.
        
        Uses GPT-4 Turbo pricing:
        - Input: $0.01 per 1K tokens
        - Output: $0.03 per 1K tokens
        
        Note: This provides a rough estimate. For accurate costs,
        track input and output tokens separately.
        
        Args:
            token_count: Total number of tokens used
        
        Returns:
            Dictionary with estimated costs in USD
        """
        # GPT-4 Turbo pricing (as of late 2024)
        # Using average of input/output pricing for estimate
        avg_cost_per_1k = 0.02  # Average of $0.01 input + $0.03 output
        
        estimated_cost = (token_count / 1000) * avg_cost_per_1k
        
        return {
            "token_count": token_count,
            "estimated_cost_usd": estimated_cost,
            "pricing_note": "Estimate based on GPT-4 Turbo average pricing"
        }


def format_sources_for_display(sources: List[Document]) -> str:
    """
    Format source documents for human-readable display.
    
    Args:
        sources: List of source Document objects
    
    Returns:
        Formatted string showing source information
    """
    if not sources:
        return "No sources available."
    
    lines = ["\n--- Sources ---"]
    for i, doc in enumerate(sources, start=1):
        metadata = doc.metadata
        source_info = metadata.get("source", "Unknown")
        score = metadata.get("similarity_score", "N/A")
        
        lines.append(f"\n[{i}] Source: {source_info}")
        if score != "N/A":
            lines.append(f"    Similarity: {score:.4f}")
        
        # Show snippet of content
        content_preview = doc.page_content[:200].replace("\n", " ")
        lines.append(f"    Preview: {content_preview}...")
    
    return "\n".join(lines)


if __name__ == "__main__":
    """
    Example usage demonstrating the BaselineRAG system.
    
    This example shows how to:
    1. Initialize a vector store
    2. Create a BaselineRAG instance
    3. Query the system
    4. Process and display results
    """
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set it in your .env file or environment")
        exit(1)
    
    print("=" * 60)
    print("Baseline RAG System Demo")
    print("=" * 60)
    
    try:
        # Import vector store
        from src.vector_store import get_vector_store
        
        # Initialize ChromaDB vector store (local development)
        print("\n[1] Initializing vector store...")
        vector_store = get_vector_store(mode="chroma")
        
        # Check if there are documents in the store
        stats = vector_store.get_collection_stats()
        print(f"    Collection: {stats['name']}")
        print(f"    Documents: {stats['count']}")
        
        if stats['count'] == 0:
            print("\n⚠️  Warning: No documents in vector store!")
            print("    Run the ingestion script first:")
            print("    python scripts/process_documents.py")
            exit(0)
        
        # Initialize BaselineRAG
        print("\n[2] Initializing BaselineRAG...")
        rag = BaselineRAG(vector_store=vector_store, k=5)
        print(f"    Model: {rag.MODEL_NAME}")
        print(f"    Temperature: {rag.TEMPERATURE}")
        print(f"    Retrieved docs (k): {rag.k}")
        
        # Example queries
        example_questions = [
            "What were the main revenue drivers mentioned in the financial reports?",
            "What risks did the company identify in their SEC filings?",
            "How did the company's profit margin change year over year?"
        ]
        
        print("\n[3] Running example queries...")
        print("-" * 60)
        
        for i, question in enumerate(example_questions, start=1):
            print(f"\nQuery {i}: {question}")
            print("-" * 40)
            
            # Execute query
            result = rag.query(question)
            
            # Display results
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nMetrics:")
            print(f"  - Latency: {result['latency']:.2f} seconds")
            print(f"  - Tokens used: {result['token_count']}")
            
            # Cost estimate
            cost_info = rag.get_cost_estimate(result['token_count'])
            print(f"  - Estimated cost: ${cost_info['estimated_cost_usd']:.4f}")
            
            # Show sources
            print(format_sources_for_display(result['sources']))
            
            print("\n" + "=" * 60)
        
        print("\nDemo completed successfully!")
        
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        
    except Exception as e:
        print(f"\nError: {e}")
        raise

