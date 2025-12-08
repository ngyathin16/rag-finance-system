"""
Self-Correcting RAG Orchestrator for Financial Document Question Answering.

This module provides a SelfCorrectingRAG class that orchestrates the retrieval,
relevance filtering, generation, and fact-checking agents to produce verified,
citation-backed answers to financial questions.

Features:
    - Multi-stage RAG pipeline with self-correction loop
    - Automatic fact-checking and answer regeneration
    - Comprehensive OpenTelemetry tracing for observability
    - Detailed metrics: latency, tokens, costs, corrections
    - Configurable correction attempts and thresholds

Usage:
    from src.vector_store import get_vector_store
    from src.orchestrator import SelfCorrectingRAG
    
    vector_store = get_vector_store(mode="chroma")
    rag = SelfCorrectingRAG(vector_store=vector_store, max_corrections=2)
    
    result = rag.query("What was the company's revenue in Q4 2023?")
    print(result["answer"])
    print(f"Verified: {result['verification_status']}")
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry import trace

from src.agents.fact_check_agent import FactCheckAgent, FactCheckResult
from src.agents.generator_agent import GeneratorAgent
from src.agents.relevance_agent import RelevanceAgent
from src.vector_store import BaseVectorStore
from src.observability.tracing import setup_tracing, get_tracer
from src.observability.metrics import MetricsCollector, get_metrics_collector

# Configure logging
logger = logging.getLogger(__name__)

# Initialize tracing at module load
_tracer_initialized = False

def _ensure_tracing_initialized() -> None:
    """Ensure tracing is initialized once."""
    global _tracer_initialized
    if not _tracer_initialized:
        try:
            setup_tracing(service_name="rag-finance-system")
            _tracer_initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize tracing: {e}")

# Get tracer for OpenTelemetry spans
tracer = trace.get_tracer(__name__)


class InsufficientDocumentsError(Exception):
    """Raised when insufficient relevant documents are found for a query."""
    
    def __init__(self, message: str, docs_retrieved: int = 0, docs_after_filter: int = 0):
        """
        Initialize the exception.
        
        Args:
            message: Error description
            docs_retrieved: Number of documents initially retrieved
            docs_after_filter: Number of documents after relevance filtering
        """
        super().__init__(message)
        self.docs_retrieved = docs_retrieved
        self.docs_after_filter = docs_after_filter


class SelfCorrectingRAG:
    """
    Self-correcting RAG system with fact-checking and automatic regeneration.
    
    This orchestrator implements a multi-stage RAG pipeline:
    1. Retrieval: Fetch documents from vector store
    2. Relevance Filtering: Score and filter documents by relevance
    3. Generation Loop: Generate answer with self-correction
       - Generate citation-backed answer
       - Fact-check against sources
       - Retry with additional context if needed
    4. Return verified answer with comprehensive metrics
    
    The self-correction mechanism ensures higher quality answers by detecting
    and correcting hallucinations, misquoted numbers, and unsupported claims.
    
    Attributes:
        vector_store: The vector store containing document embeddings
        max_corrections: Maximum number of correction attempts allowed
        relevance_agent: Agent for scoring document relevance
        generator_agent: Agent for generating citation-backed answers
        fact_check_agent: Agent for verifying answer accuracy
    
    Example:
        >>> from src.vector_store import get_vector_store
        >>> store = get_vector_store(mode="chroma")
        >>> rag = SelfCorrectingRAG(vector_store=store, max_corrections=2)
        >>> result = rag.query("What was the revenue growth in Q4 2023?")
        >>> print(f"Answer: {result['answer']}")
        >>> print(f"Confidence: {result['confidence']:.2f}")
        >>> print(f"Verified: {result['verification_status']}")
        >>> print(f"Corrections made: {result['corrections_made']}")
    """
    
    # Default configuration
    INITIAL_K = 12  # Number of documents to retrieve initially
    RELEVANCE_THRESHOLD = 0.7  # Minimum relevance score for filtering
    MIN_DOCS_REQUIRED = 1  # Minimum documents needed to generate answer
    
    # Model configuration
    RELEVANCE_MODEL = "gpt-4o-mini"
    GENERATOR_MODEL = "gpt-4-turbo-preview"
    FACT_CHECK_MODEL = "gpt-4o-mini"
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        max_corrections: int = 2,
        enable_metrics: bool = True
    ):
        """
        Initialize the SelfCorrectingRAG orchestrator.
        
        Args:
            vector_store: A BaseVectorStore instance containing document embeddings.
                         Can be ChromaVectorStore for local development or
                         PineconeVectorStore for production.
            max_corrections: Maximum number of correction attempts before
                            returning the best available answer (default: 2)
            enable_metrics: Whether to enable metrics collection (default: True)
        
        Raises:
            ValueError: If vector_store is None or max_corrections is negative
        """
        if vector_store is None:
            raise ValueError("vector_store cannot be None")
        
        if max_corrections < 0:
            raise ValueError("max_corrections must be non-negative")
        
        self.vector_store = vector_store
        self.max_corrections = max_corrections
        self.enable_metrics = enable_metrics
        
        # Initialize tracing
        _ensure_tracing_initialized()
        
        # Initialize metrics collector
        self.metrics_collector: Optional[MetricsCollector] = None
        if enable_metrics:
            try:
                self.metrics_collector = get_metrics_collector(
                    service_name="rag-finance-system"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize metrics collector: {e}")
        
        # Initialize agents
        self.relevance_agent = RelevanceAgent(model_name=self.RELEVANCE_MODEL)
        self.generator_agent = GeneratorAgent(model_name=self.GENERATOR_MODEL)
        self.fact_check_agent = FactCheckAgent(model_name=self.FACT_CHECK_MODEL)
        
        logger.info(
            f"SelfCorrectingRAG initialized: max_corrections={max_corrections}, "
            f"initial_k={self.INITIAL_K}, relevance_threshold={self.RELEVANCE_THRESHOLD}, "
            f"metrics_enabled={enable_metrics}"
        )
    
    def _add_correction_context(
        self,
        docs: List[Tuple[str, float, str]],
        unsupported_claims: List[str]
    ) -> List[Tuple[str, float, str]]:
        """
        Add correction context to documents based on unsupported claims.
        
        This method enhances the document context by adding explicit notes
        about claims that need verification, helping the generator produce
        more accurate answers in subsequent attempts.
        
        Args:
            docs: List of (content, relevance_score, reasoning) tuples
            unsupported_claims: List of claims that were not supported
                               in the previous generation attempt
        
        Returns:
            Enhanced list of (content, relevance_score, reasoning) tuples
            with correction context prepended
        """
        if not unsupported_claims:
            return docs
        
        # Create correction context as a high-priority "document"
        correction_note = (
            "CORRECTION CONTEXT: The following claims from the previous answer "
            "were NOT supported by sources and should be avoided or corrected:\n"
            + "\n".join(f"- {claim}" for claim in unsupported_claims)
            + "\n\nPlease generate a new answer using ONLY information from the sources below."
        )
        
        # Add correction context as first document with high relevance
        correction_doc = (correction_note, 1.0, "Correction guidance from fact-checker")
        
        return [correction_doc] + list(docs)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the self-correcting RAG system with a financial question.
        
        This method executes the full RAG pipeline with self-correction:
        
        1. **Retrieval**: Fetch k=12 documents from vector store
        2. **Relevance Filtering**: Score documents and filter by threshold=0.7
        3. **Generation Loop**:
           a. Generate answer with citations
           b. Fact-check answer against sources
           c. If correction needed and attempts remaining:
              - Log correction trigger
              - Add correction context
              - Retry generation
           d. Return when verified or max corrections reached
        
        Args:
            question: The financial question to answer
        
        Returns:
            Dictionary containing:
                - answer (str): The generated answer with citations
                - confidence (float): Confidence score for the answer
                - verification_status (str): "VERIFIED", "UNCERTAIN", or "FALSE"
                - sources (List[Tuple]): Source documents with scores
                - corrections_made (int): Number of correction attempts
                - latency (float): Total processing time in seconds
                - total_tokens (int): Total tokens used across all calls
                - total_cost_usd (float): Estimated total cost in USD
                - warning (Optional[str]): Warning if max corrections reached
        
        Raises:
            ValueError: If question is empty
            InsufficientDocumentsError: If not enough relevant documents found
        
        Example:
            >>> result = rag.query("What was the company's Q4 2023 revenue?")
            >>> print(result["answer"])
            "The company reported Q4 2023 revenue of $4.2 billion [1]..."
            >>> print(f"Status: {result['verification_status']}")
            "Status: VERIFIED"
        """
        if not question or not question.strip():
            raise ValueError("question cannot be empty")
        
        # Track overall timing
        start_time = time.perf_counter()
        
        # Track cumulative metrics
        total_tokens = 0
        total_cost_usd = 0.0
        corrections_made = 0
        warning_message: Optional[str] = None
        
        # Start main OpenTelemetry span
        with tracer.start_as_current_span("rag_query") as main_span:
            main_span.set_attribute("question", question[:200])  # Truncate for span
            main_span.set_attribute("max_corrections", self.max_corrections)
            
            logger.info(f"Query received: {question[:100]}...")
            
            # ================================================================
            # Step 1: Retrieve documents from vector store
            # ================================================================
            with tracer.start_as_current_span("retrieval") as retrieval_span:
                retrieval_span.set_attribute("k", self.INITIAL_K)
                
                try:
                    # Perform similarity search
                    raw_results = self.vector_store.similarity_search(
                        query=question,
                        k=self.INITIAL_K
                    )
                    
                    # Convert to format expected by relevance agent: (content, score)
                    documents_for_scoring: List[Tuple[str, float]] = [
                        (doc.page_content, score)
                        for doc, score in raw_results
                    ]
                    
                    retrieval_span.set_attribute("docs_retrieved", len(documents_for_scoring))
                    logger.info(f"Documents retrieved: {len(documents_for_scoring)}")
                    
                except Exception as e:
                    retrieval_span.set_attribute("error", str(e))
                    logger.error(f"Retrieval failed: {e}")
                    raise
            
            # ================================================================
            # Step 2: Relevance filtering
            # ================================================================
            with tracer.start_as_current_span("relevance_filtering") as filter_span:
                filter_span.set_attribute("threshold", self.RELEVANCE_THRESHOLD)
                filter_span.set_attribute("input_docs", len(documents_for_scoring))
                
                try:
                    # Score and filter documents
                    scored_documents = self.relevance_agent.score_documents(
                        query=question,
                        documents=documents_for_scoring,
                        threshold=self.RELEVANCE_THRESHOLD
                    )
                    
                    filter_span.set_attribute("docs_after_filter", len(scored_documents))
                    logger.info(
                        f"Documents filtered: {len(documents_for_scoring)} -> "
                        f"{len(scored_documents)} (threshold={self.RELEVANCE_THRESHOLD})"
                    )
                    
                    # Check minimum document requirement
                    if len(scored_documents) < self.MIN_DOCS_REQUIRED:
                        filter_span.set_attribute("error", "insufficient_documents")
                        raise InsufficientDocumentsError(
                            f"Insufficient relevant documents found. "
                            f"Retrieved {len(documents_for_scoring)}, "
                            f"but only {len(scored_documents)} passed relevance threshold "
                            f"(minimum required: {self.MIN_DOCS_REQUIRED})",
                            docs_retrieved=len(documents_for_scoring),
                            docs_after_filter=len(scored_documents)
                        )
                    
                except InsufficientDocumentsError:
                    raise
                except Exception as e:
                    filter_span.set_attribute("error", str(e))
                    logger.error(f"Relevance filtering failed: {e}")
                    raise
            
            # ================================================================
            # Step 3: Generation loop with self-correction
            # ================================================================
            with tracer.start_as_current_span("generation_loop") as gen_loop_span:
                gen_loop_span.set_attribute("max_attempts", self.max_corrections + 1)
                
                # Initialize for generation loop
                current_docs = scored_documents
                best_answer: Optional[Dict[str, Any]] = None
                best_verification: Optional[FactCheckResult] = None
                attempt = 0
                
                while attempt <= self.max_corrections:
                    attempt += 1
                    logger.info(f"Generation attempt {attempt}/{self.max_corrections + 1}")
                    
                    # ----- Generate answer -----
                    try:
                        gen_result = self.generator_agent.generate(
                            query=question,
                            scored_documents=current_docs
                        )
                        
                        # Track tokens and cost
                        total_tokens += gen_result.get("token_count", 0)
                        total_cost_usd += gen_result.get("cost_usd", 0.0)
                        
                        logger.info(
                            f"Answer generated (attempt {attempt}): "
                            f"confidence={gen_result['confidence']:.2f}"
                        )
                        
                    except Exception as e:
                        logger.error(f"Generation failed (attempt {attempt}): {e}")
                        gen_loop_span.set_attribute("generation_error", str(e))
                        raise
                    
                    # ----- Fact-check answer -----
                    with tracer.start_as_current_span("fact_checking") as fc_span:
                        fc_span.set_attribute("attempt", attempt)
                        
                        try:
                            # Use original scored_documents for fact-checking
                            # (not including correction context)
                            verification = self.fact_check_agent.verify(
                                answer=gen_result["answer"],
                                sources=scored_documents
                            )
                            
                            fc_span.set_attribute("status", verification.status)
                            fc_span.set_attribute("correction_needed", verification.correction_needed)
                            fc_span.set_attribute("confidence", verification.confidence)
                            fc_span.set_attribute(
                                "unsupported_claims_count",
                                len(verification.unsupported_claims)
                            )
                            
                            logger.info(
                                f"Verification status: {verification.status}, "
                                f"correction_needed={verification.correction_needed}, "
                                f"unsupported_claims={len(verification.unsupported_claims)}"
                            )
                            
                        except Exception as e:
                            logger.error(f"Fact-checking failed (attempt {attempt}): {e}")
                            fc_span.set_attribute("error", str(e))
                            raise
                    
                    # Store best result so far
                    best_answer = gen_result
                    best_verification = verification
                    
                    # ----- Check if we're done -----
                    if not verification.correction_needed:
                        # Answer verified, we're done
                        logger.info(f"Answer verified on attempt {attempt}")
                        break
                    
                    # ----- Handle correction -----
                    if attempt <= self.max_corrections:
                        corrections_made += 1
                        logger.info(
                            f"Correction triggered (attempt {attempt}): "
                            f"unsupported_claims={verification.unsupported_claims}"
                        )
                        
                        # Add correction context for next attempt
                        current_docs = self._add_correction_context(
                            scored_documents,
                            verification.unsupported_claims
                        )
                    else:
                        # Max corrections reached
                        warning_message = (
                            f"Maximum corrections ({self.max_corrections}) reached. "
                            f"Returning best available answer with status: {verification.status}"
                        )
                        logger.warning(warning_message)
                
                # Set generation loop metrics
                gen_loop_span.set_attribute("total_attempts", attempt)
                gen_loop_span.set_attribute("corrections_made", corrections_made)
                gen_loop_span.set_attribute(
                    "final_status",
                    best_verification.status if best_verification else "UNKNOWN"
                )
            
            # ================================================================
            # Calculate final metrics and build response
            # ================================================================
            latency = time.perf_counter() - start_time
            
            # Set main span attributes
            main_span.set_attribute("latency_seconds", latency)
            main_span.set_attribute("total_tokens", total_tokens)
            main_span.set_attribute("total_cost_usd", total_cost_usd)
            main_span.set_attribute("corrections_made", corrections_made)
            main_span.set_attribute(
                "verification_status",
                best_verification.status if best_verification else "UNKNOWN"
            )
            
            # Prepare sources as list of tuples for response
            sources_for_response: List[Tuple[str, float, str]] = scored_documents
            
            logger.info(
                f"Final result: verification_status={best_verification.status if best_verification else 'UNKNOWN'}, "
                f"confidence={best_answer['confidence'] if best_answer else 0:.2f}, "
                f"corrections_made={corrections_made}, "
                f"latency={latency:.2f}s, "
                f"total_tokens={total_tokens}, "
                f"total_cost=${total_cost_usd:.4f}"
            )
            
            # Build response dictionary
            response: Dict[str, Any] = {
                "answer": best_answer["answer"] if best_answer else "",
                "confidence": best_answer["confidence"] if best_answer else 0.0,
                "verification_status": best_verification.status if best_verification else "UNKNOWN",
                "sources": sources_for_response,
                "corrections_made": corrections_made,
                "latency": latency,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost_usd,
            }
            
            # Add warning if max corrections reached
            if warning_message:
                response["warning"] = warning_message
            
            # Record metrics
            if self.metrics_collector:
                try:
                    # Calculate average relevance score from sources
                    avg_relevance = (
                        sum(score for _, score, _ in sources_for_response) / len(sources_for_response)
                        if sources_for_response else 0.0
                    )
                    
                    self.metrics_collector.record_query(
                        query=question,
                        latency_seconds=latency,
                        tokens_used=total_tokens,
                        cost_usd=total_cost_usd,
                        corrections=corrections_made,
                        relevance_score=avg_relevance,
                        verification_status=best_verification.status if best_verification else "UNKNOWN",
                        answer=best_answer["answer"] if best_answer else None,
                        source_count=len(sources_for_response),
                        metadata={
                            "confidence": best_answer["confidence"] if best_answer else 0.0,
                            "max_corrections": self.max_corrections,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to record metrics: {e}")
            
            return response


if __name__ == "__main__":
    """
    Example usage demonstrating the SelfCorrectingRAG orchestrator.
    
    This example shows how to:
    1. Initialize the vector store and orchestrator
    2. Query the system with a financial question
    3. Process and display results with metrics
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
    
    print("=" * 70)
    print("Self-Correcting RAG Orchestrator Demo")
    print("=" * 70)
    
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
        
        # Initialize SelfCorrectingRAG
        print("\n[2] Initializing SelfCorrectingRAG orchestrator...")
        rag = SelfCorrectingRAG(vector_store=vector_store, max_corrections=2)
        print(f"    Max corrections: {rag.max_corrections}")
        print(f"    Initial k: {rag.INITIAL_K}")
        print(f"    Relevance threshold: {rag.RELEVANCE_THRESHOLD}")
        
        # Example queries
        example_questions = [
            "What was the company's total revenue and how did it compare to the previous year?",
            "What are the main risk factors mentioned in the financial documents?",
            "How did operating expenses change and what drove those changes?"
        ]
        
        print("\n[3] Running example queries...")
        print("-" * 70)
        
        for i, question in enumerate(example_questions, start=1):
            print(f"\n{'='*70}")
            print(f"Query {i}: {question}")
            print("=" * 70)
            
            try:
                # Execute query
                result = rag.query(question)
                
                # Display results
                print(f"\n📝 Answer:\n{result['answer']}")
                
                print(f"\n📊 Metrics:")
                print(f"  • Confidence: {result['confidence']:.2f}")
                print(f"  • Verification Status: {result['verification_status']}")
                print(f"  • Corrections Made: {result['corrections_made']}")
                print(f"  • Latency: {result['latency']:.2f} seconds")
                print(f"  • Total Tokens: {result['total_tokens']}")
                print(f"  • Total Cost: ${result['total_cost_usd']:.4f}")
                
                if "warning" in result:
                    print(f"\n⚠️  Warning: {result['warning']}")
                
                print(f"\n📚 Sources ({len(result['sources'])} documents):")
                for j, (content, score, reasoning) in enumerate(result['sources'][:3], start=1):
                    print(f"\n  [{j}] Relevance: {score:.2f}")
                    print(f"      Reasoning: {reasoning}")
                    print(f"      Content: {content[:150]}...")
                
            except InsufficientDocumentsError as e:
                print(f"\n❌ Insufficient documents: {e}")
                print(f"   Retrieved: {e.docs_retrieved}")
                print(f"   After filter: {e.docs_after_filter}")
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        print("\n" + "=" * 70)
        print("Demo completed!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        
    except Exception as e:
        print(f"\nError: {e}")
        raise

