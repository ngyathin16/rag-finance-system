"""
Vector Store Manager for RAG Finance System.

Provides two implementations:
- ChromaVectorStore: Local development with ChromaDB
- PineconeVectorStore: Production with Pinecone serverless

Usage:
    from src.vector_store import get_vector_store
    
    # Local development
    store = get_vector_store(mode="chroma")
    
    # Production
    store = get_vector_store(mode="pinecone")
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingCostTracker:
    """Tracks embedding generation costs and counts."""
    
    # Pricing for text-embedding-3-small (per 1M tokens)
    COST_PER_1M_TOKENS = 0.02
    
    def __init__(self):
        self.total_embeddings = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def log_embeddings(self, count: int, estimated_tokens: Optional[int] = None) -> None:
        """
        Log embedding generation.
        
        Args:
            count: Number of embeddings generated
            estimated_tokens: Estimated token count (if available)
        """
        self.total_embeddings += count
        if estimated_tokens:
            self.total_tokens += estimated_tokens
            cost = (estimated_tokens / 1_000_000) * self.COST_PER_1M_TOKENS
            self.total_cost += cost
        
        logger.info(
            f"Embeddings generated: {count} | "
            f"Total embeddings: {self.total_embeddings} | "
            f"Estimated cost: ${self.total_cost:.6f}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        return {
            "total_embeddings": self.total_embeddings,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost
        }


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""
    
    def __init__(self):
        self.cost_tracker = EmbeddingCostTracker()
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    @abstractmethod
    def add_documents(
        self, 
        documents: List[Document], 
        batch_size: int = 100
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects to add
            batch_size: Number of documents to process per batch
        """
        pass
    
    @abstractmethod
    def similarity_search(
        self, 
        query: str, 
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples, sorted by relevance
        """
        pass
    
    @abstractmethod
    def hybrid_search(
        self, 
        query: str, 
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining vector similarity and keyword matching.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples, sorted by relevance
        """
        pass
    
    def _estimate_tokens(self, texts: List[str]) -> int:
        """Estimate token count for a list of texts (rough approximation)."""
        # Approximate: 1 token ≈ 4 characters for English text
        total_chars = sum(len(text) for text in texts)
        return total_chars // 4


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store for local development.
    
    Features:
    - Persistent storage to data/chroma_db/
    - Collection name: financial_docs
    - Supports similarity and hybrid search
    
    Example:
        store = ChromaVectorStore()
        store.add_documents(documents)
        results = store.similarity_search("revenue growth", k=5)
    """
    
    COLLECTION_NAME = "financial_docs"
    PERSIST_DIRECTORY = "data/chroma_db"
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data.
                             Defaults to data/chroma_db/
        """
        super().__init__()
        
        self._persist_directory = persist_directory or self.PERSIST_DIRECTORY
        
        # Ensure directory exists
        os.makedirs(self._persist_directory, exist_ok=True)
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Initialize Chroma client with persistence
            self._client = chromadb.PersistentClient(
                path=self._persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(
                f"ChromaDB initialized: collection='{self.COLLECTION_NAME}', "
                f"persist_dir='{self._persist_directory}'"
            )
            
        except ImportError:
            raise ImportError(
                "chromadb package is required. Install with: pip install chromadb"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(
        self, 
        documents: List[Document], 
        batch_size: int = 100
    ) -> None:
        """
        Add documents to ChromaDB in batches.
        
        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents per batch (default: 100)
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to ChromaDB")
        
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(documents), batch_size), 
                      desc="Adding documents", 
                      total=total_batches):
            batch = documents[i:i + batch_size]
            
            try:
                # Extract texts and metadata
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                ids = [
                    doc.metadata.get("id", f"doc_{i + j}") 
                    for j, doc in enumerate(batch)
                ]
                
                # Generate embeddings
                embeddings = self._embeddings.embed_documents(texts)
                
                # Track costs
                estimated_tokens = self._estimate_tokens(texts)
                self.cost_tracker.log_embeddings(len(batch), estimated_tokens)
                
                # Add to collection
                self._collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
            except Exception as e:
                logger.error(f"Failed to add batch {i // batch_size + 1}: {e}")
                raise RuntimeError(f"Failed to add documents to ChromaDB: {e}")
        
        logger.info(f"Successfully added {len(documents)} documents")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Perform cosine similarity search.
        
        Args:
            query: Search query
            k: Number of results (default: 10)
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self._embeddings.embed_query(query)
            self.cost_tracker.log_embeddings(1, self._estimate_tokens([query]))
            
            # Query collection
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to Document objects with scores
            documents_with_scores = []
            
            if results["documents"] and results["documents"][0]:
                for doc_text, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    doc = Document(page_content=doc_text, metadata=metadata or {})
                    # Convert distance to similarity score (cosine)
                    score = 1 - distance
                    documents_with_scores.append((doc, score))
            
            logger.debug(f"Similarity search returned {len(documents_with_scores)} results")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RuntimeError(f"Similarity search failed: {e}")
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining vector similarity and keyword matching.
        
        ChromaDB doesn't natively support hybrid search, so we implement
        a simple combination of vector search and keyword filtering.
        
        Args:
            query: Search query
            k: Number of results (default: 10)
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            # Get more results for re-ranking
            vector_results = self.similarity_search(query, k=k * 2)
            
            # Simple keyword boosting
            query_terms = set(query.lower().split())
            boosted_results = []
            
            for doc, score in vector_results:
                # Check keyword overlap
                doc_terms = set(doc.page_content.lower().split())
                overlap = len(query_terms & doc_terms)
                
                # Boost score based on keyword overlap
                boost = 1 + (overlap * 0.1)
                boosted_score = min(score * boost, 1.0)
                
                boosted_results.append((doc, boosted_score))
            
            # Sort by boosted score and return top k
            boosted_results.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Hybrid search returned {len(boosted_results[:k])} results")
            return boosted_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise RuntimeError(f"Hybrid search failed: {e}")
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
            logger.info(f"Deleted collection: {self.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            "name": self.COLLECTION_NAME,
            "count": self._collection.count(),
            "persist_directory": self._persist_directory
        }


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone vector store for production deployment.
    
    Features:
    - Serverless index with automatic scaling
    - Metadata filtering support
    - Dimension: 1536 (text-embedding-3-small)
    - Metric: cosine
    
    Environment Variables:
        PINECONE_API_KEY: Pinecone API key
        PINECONE_INDEX_NAME: Index name (default: financial-docs)
        PINECONE_CLOUD: Cloud provider (default: aws)
        PINECONE_REGION: Region (default: us-east-1)
    
    Example:
        store = PineconeVectorStore()
        store.add_documents(documents)
        results = store.similarity_search("revenue", k=5, filter={"source": "10-K"})
    """
    
    EMBEDDING_DIMENSION = 1536
    DEFAULT_INDEX_NAME = "financial-docs"
    DEFAULT_CLOUD = "aws"
    DEFAULT_REGION = "us-east-1"
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None
    ):
        """
        Initialize Pinecone vector store.
        
        Args:
            index_name: Pinecone index name (default from env or 'financial-docs')
            cloud: Cloud provider (default: aws)
            region: Region (default: us-east-1)
        """
        super().__init__()
        
        self._api_key = os.getenv("PINECONE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable is required"
            )
        
        self._index_name = (
            index_name or 
            os.getenv("PINECONE_INDEX_NAME", self.DEFAULT_INDEX_NAME)
        )
        self._cloud = cloud or os.getenv("PINECONE_CLOUD", self.DEFAULT_CLOUD)
        self._region = region or os.getenv("PINECONE_REGION", self.DEFAULT_REGION)
        
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            # Initialize Pinecone client
            self._pc = Pinecone(api_key=self._api_key)
            
            # Create index if it doesn't exist
            existing_indexes = [idx.name for idx in self._pc.list_indexes()]
            
            if self._index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self._index_name}")
                self._pc.create_index(
                    name=self._index_name,
                    dimension=self.EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self._cloud,
                        region=self._region
                    )
                )
                logger.info(f"Created serverless index: {self._index_name}")
            
            # Connect to index
            self._index = self._pc.Index(self._index_name)
            
            logger.info(
                f"Pinecone initialized: index='{self._index_name}', "
                f"cloud='{self._cloud}', region='{self._region}'"
            )
            
        except ImportError:
            raise ImportError(
                "pinecone-client package is required. "
                "Install with: pip install pinecone-client"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def add_documents(
        self, 
        documents: List[Document], 
        batch_size: int = 100
    ) -> None:
        """
        Add documents to Pinecone in batches.
        
        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents per batch (default: 100)
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to Pinecone")
        
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(documents), batch_size), 
                      desc="Adding documents", 
                      total=total_batches):
            batch = documents[i:i + batch_size]
            
            try:
                # Extract texts
                texts = [doc.page_content for doc in batch]
                
                # Generate embeddings
                embeddings = self._embeddings.embed_documents(texts)
                
                # Track costs
                estimated_tokens = self._estimate_tokens(texts)
                self.cost_tracker.log_embeddings(len(batch), estimated_tokens)
                
                # Prepare vectors for upsert
                vectors = []
                for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
                    vector_id = doc.metadata.get("id", f"doc_{i + j}")
                    
                    # Prepare metadata (Pinecone has size limits)
                    metadata = {
                        **doc.metadata,
                        "text": doc.page_content[:1000]  # Truncate for metadata
                    }
                    
                    vectors.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    })
                
                # Upsert to Pinecone
                self._index.upsert(vectors=vectors)
                
            except Exception as e:
                logger.error(f"Failed to add batch {i // batch_size + 1}: {e}")
                raise RuntimeError(f"Failed to add documents to Pinecone: {e}")
        
        logger.info(f"Successfully added {len(documents)} documents")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with optional metadata filtering.
        
        Args:
            query: Search query
            k: Number of results (default: 10)
            filter: Metadata filter dictionary (optional)
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self._embeddings.embed_query(query)
            self.cost_tracker.log_embeddings(1, self._estimate_tokens([query]))
            
            # Query Pinecone
            query_params = {
                "vector": query_embedding,
                "top_k": k,
                "include_metadata": True
            }
            
            if filter:
                query_params["filter"] = filter
            
            results = self._index.query(**query_params)
            
            # Convert to Document objects with scores
            documents_with_scores = []
            
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                text = metadata.pop("text", "")
                
                doc = Document(page_content=text, metadata=metadata)
                score = match.get("score", 0.0)
                documents_with_scores.append((doc, score))
            
            logger.debug(f"Similarity search returned {len(documents_with_scores)} results")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RuntimeError(f"Similarity search failed: {e}")
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining dense and sparse vectors.
        
        Note: For true hybrid search with Pinecone, you would need to use
        sparse-dense vectors. This implementation provides a simpler
        keyword-boosted approach.
        
        Args:
            query: Search query
            k: Number of results (default: 10)
            filter: Metadata filter dictionary (optional)
            alpha: Balance between vector (0) and keyword (1) search
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            # Get vector search results
            vector_results = self.similarity_search(query, k=k * 2, filter=filter)
            
            # Apply keyword boosting
            query_terms = set(query.lower().split())
            boosted_results = []
            
            for doc, score in vector_results:
                doc_terms = set(doc.page_content.lower().split())
                overlap = len(query_terms & doc_terms)
                
                # Weighted combination
                keyword_score = min(overlap / len(query_terms), 1.0) if query_terms else 0
                hybrid_score = (1 - alpha) * score + alpha * keyword_score
                
                boosted_results.append((doc, hybrid_score))
            
            # Sort by hybrid score
            boosted_results.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Hybrid search returned {len(boosted_results[:k])} results")
            return boosted_results[:k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise RuntimeError(f"Hybrid search failed: {e}")
    
    def delete_all(self) -> None:
        """Delete all vectors from the index."""
        try:
            self._index.delete(delete_all=True)
            logger.info(f"Deleted all vectors from index: {self._index_name}")
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            stats = self._index.describe_index_stats()
            return {
                "name": self._index_name,
                "dimension": stats.get("dimension"),
                "total_vector_count": stats.get("total_vector_count"),
                "namespaces": stats.get("namespaces", {})
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise


def get_vector_store(mode: str = "chroma", **kwargs) -> BaseVectorStore:
    """
    Factory function to get the appropriate vector store implementation.
    
    Args:
        mode: Either "chroma" for local development or "pinecone" for production
        **kwargs: Additional arguments passed to the vector store constructor
        
    Returns:
        Vector store instance (ChromaVectorStore or PineconeVectorStore)
        
    Raises:
        ValueError: If mode is not "chroma" or "pinecone"
        
    Example:
        # Local development
        store = get_vector_store(mode="chroma")
        
        # Production with custom index name
        store = get_vector_store(mode="pinecone", index_name="my-index")
    """
    mode = mode.lower()
    
    if mode == "chroma":
        logger.info("Initializing ChromaDB vector store (local development)")
        return ChromaVectorStore(**kwargs)
    
    elif mode == "pinecone":
        logger.info("Initializing Pinecone vector store (production)")
        return PineconeVectorStore(**kwargs)
    
    else:
        raise ValueError(
            f"Invalid mode: '{mode}'. Must be 'chroma' or 'pinecone'"
        )


# Convenience exports
__all__ = [
    "BaseVectorStore",
    "ChromaVectorStore", 
    "PineconeVectorStore",
    "get_vector_store",
    "EmbeddingCostTracker"
]

