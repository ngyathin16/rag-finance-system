"""
Relevance Agent for Financial Document Scoring.

This module provides a RelevanceAgent that uses LLMs with structured output
to score and filter documents based on their relevance to a given query.
Specifically designed for financial Q&A use cases with strict relevance criteria.

Features:
    - LLM-based relevance scoring with structured Pydantic output
    - Configurable threshold filtering
    - OpenTelemetry instrumentation for observability
    - Batch processing of documents
    - Sorted results by relevance score

Usage:
    from src.agents.relevance_agent import RelevanceAgent
    
    agent = RelevanceAgent(model_name="gpt-4o-mini")
    
    documents = [
        ("Revenue grew by 15% in Q4 2023...", 0.85),
        ("The weather forecast shows rain...", 0.72),
    ]
    
    filtered = agent.score_documents(
        query="What was the revenue growth?",
        documents=documents,
        threshold=0.7
    )
"""

import logging
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from opentelemetry import trace
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Get tracer for OpenTelemetry spans
tracer = trace.get_tracer(__name__)


class RelevanceScore(BaseModel):
    """
    Structured output model for document relevance scoring.
    
    This model enforces a consistent format for LLM responses when
    evaluating document relevance to a query.
    
    Attributes:
        score: Relevance score between 0.0 and 1.0
        reasoning: Brief explanation for the assigned score
    
    Example:
        >>> score = RelevanceScore(
        ...     score=0.85,
        ...     reasoning="Document contains specific Q4 2023 revenue figures"
        ... )
    """
    
    score: float = Field(
        description="Relevance score from 0 to 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation for the relevance score",
        max_length=200
    )


# System prompt with strict relevance criteria for financial Q&A
RELEVANCE_SYSTEM_PROMPT = """You are a financial document relevance scorer. Your task is to evaluate how relevant a document is to answering a specific financial question.

Use the following scoring criteria strictly:

**1.0 (Perfect Match):** Document provides a direct answer with specific data points, numbers, or facts that directly address the question.

**0.7-0.9 (Highly Relevant):** Document contains relevant context and information that helps answer the question, but may lack specific data points or require some inference.

**0.4-0.6 (Tangentially Related):** Document discusses related topics or concepts but doesn't directly address the question. May provide useful background but not a direct answer.

**0.0-0.3 (Not Relevant):** Document is unrelated to the question or discusses completely different topics, time periods, or entities.

Consider these factors when scoring:
- Does the document contain the specific financial metrics, dates, or entities mentioned in the question?
- Is the information from the correct time period?
- Does the document provide actionable information to answer the question?
- Are there specific numbers, percentages, or data points that directly answer the query?

Be strict and precise in your scoring. Financial Q&A requires accurate, specific information."""

RELEVANCE_USER_PROMPT = """Question: {query}

Document to evaluate:
{document}

Score this document's relevance to the question."""


class RelevanceAgent:
    """
    Agent for scoring document relevance using LLM with structured output.
    
    This agent evaluates how relevant retrieved documents are to a user's query,
    providing both a numerical score and reasoning for each document. It uses
    LangChain's structured output capabilities to ensure consistent, parseable
    responses from the LLM.
    
    The agent is specifically tuned for financial document Q&A, with scoring
    criteria that emphasize specific data, accurate time periods, and
    actionable financial information.
    
    Attributes:
        model_name: Name of the OpenAI model to use
        llm: The ChatOpenAI language model instance
        chain: The prompt-to-structured-output chain
    
    Example:
        >>> agent = RelevanceAgent(model_name="gpt-4o-mini")
        >>> docs = [
        ...     ("Revenue increased 20% YoY in 2023", 0.9),
        ...     ("The company was founded in 1985", 0.7),
        ... ]
        >>> results = agent.score_documents(
        ...     query="What was the revenue growth?",
        ...     documents=docs,
        ...     threshold=0.7
        ... )
        >>> for content, score, reasoning in results:
        ...     print(f"Score: {score:.2f} - {reasoning}")
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the RelevanceAgent.
        
        Args:
            model_name: Name of the OpenAI model to use for relevance scoring.
                       Defaults to "gpt-4o-mini" for cost-effective scoring.
        
        Raises:
            ValueError: If model_name is empty or None
        """
        if not model_name:
            raise ValueError("model_name cannot be empty")
        
        self.model_name = model_name
        
        # Initialize LLM with low temperature for consistent scoring
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1  # Low temperature for consistent scoring
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RELEVANCE_SYSTEM_PROMPT),
            ("user", RELEVANCE_USER_PROMPT)
        ])
        
        # Build chain with structured output
        self.chain = self.prompt | self.llm.with_structured_output(RelevanceScore)
        
        logger.info(f"RelevanceAgent initialized with model: {model_name}")
    
    def score_documents(
        self,
        query: str,
        documents: List[Tuple[str, float]],
        threshold: float = 0.7
    ) -> List[Tuple[str, float, str]]:
        """
        Score documents for relevance to a query and filter by threshold.
        
        This method processes each document through the LLM to obtain a relevance
        score and reasoning. Documents are then filtered by the threshold and
        sorted by relevance score in descending order.
        
        Args:
            query: The user's question or search query
            documents: List of tuples containing (document_content, original_score).
                      The original_score is typically from vector similarity search.
            threshold: Minimum relevance score to include in results (default: 0.7)
        
        Returns:
            List of tuples containing (document_content, relevance_score, reasoning)
            for documents meeting the threshold, sorted by relevance score descending.
            Returns at most 5 documents.
        
        Raises:
            ValueError: If query is empty or threshold is out of range [0, 1]
        
        Example:
            >>> agent = RelevanceAgent()
            >>> docs = [("Financial data for Q4...", 0.85)]
            >>> results = agent.score_documents(
            ...     query="What was Q4 revenue?",
            ...     documents=docs,
            ...     threshold=0.7
            ... )
            >>> print(results[0])  # (content, score, reasoning)
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        
        # Start OpenTelemetry span
        with tracer.start_as_current_span("relevance_agent.score_documents") as span:
            # Set input attributes
            span.set_attribute("num_documents_input", len(documents))
            span.set_attribute("threshold", threshold)
            span.set_attribute("query_length", len(query))
            
            scored_documents: List[Tuple[str, float, str]] = []
            total_score = 0.0
            
            # Process each document
            for doc_content, original_score in documents:
                try:
                    # Invoke the chain to get structured relevance score
                    result: RelevanceScore = self.chain.invoke({
                        "query": query,
                        "document": doc_content
                    })
                    
                    scored_documents.append((
                        doc_content,
                        result.score,
                        result.reasoning
                    ))
                    total_score += result.score
                    
                    logger.debug(
                        f"Scored document: {result.score:.2f} - {result.reasoning[:50]}..."
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to score document: {e}")
                    # On error, use original similarity score as fallback
                    scored_documents.append((
                        doc_content,
                        original_score,
                        "Scoring failed, using original similarity score"
                    ))
                    total_score += original_score
            
            # Filter by threshold
            filtered_documents = [
                (content, score, reasoning)
                for content, score, reasoning in scored_documents
                if score >= threshold
            ]
            
            # Sort by relevance score descending
            filtered_documents.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 5
            top_documents = filtered_documents[:5]
            
            # Calculate and set output attributes
            avg_score = total_score / len(documents) if documents else 0.0
            span.set_attribute("num_documents_filtered", len(top_documents))
            span.set_attribute("avg_relevance_score", avg_score)
            span.set_attribute("num_documents_above_threshold", len(filtered_documents))
            
            logger.info(
                f"Relevance scoring complete: "
                f"input={len(documents)}, filtered={len(top_documents)}, "
                f"avg_score={avg_score:.2f}"
            )
            
            return top_documents


if __name__ == "__main__":
    """
    Example usage demonstrating the RelevanceAgent.
    
    This example shows how to:
    1. Initialize the RelevanceAgent
    2. Score a batch of documents
    3. Process and display filtered results
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
    print("Relevance Agent Demo")
    print("=" * 60)
    
    # Initialize the agent
    print("\n[1] Initializing RelevanceAgent...")
    agent = RelevanceAgent(model_name="gpt-4o-mini")
    print(f"    Model: {agent.model_name}")
    
    # Example documents (simulating vector search results)
    example_documents = [
        (
            "In Q4 2023, the company reported revenue of $4.2 billion, "
            "representing a 15% increase year-over-year. The growth was "
            "primarily driven by strong performance in cloud services.",
            0.92
        ),
        (
            "The company's headquarters is located in San Francisco, California. "
            "It was founded in 2008 and employs over 10,000 people worldwide.",
            0.78
        ),
        (
            "Risk factors include regulatory changes, competition, and market "
            "volatility. The company operates in a highly competitive environment "
            "with rapid technological changes.",
            0.75
        ),
        (
            "Operating expenses increased by 8% in 2023, primarily due to "
            "investments in R&D and sales expansion. Net income margin "
            "improved to 22% from 19% in the prior year.",
            0.88
        ),
        (
            "The weather in the region has been particularly mild this winter, "
            "with temperatures averaging 5 degrees above normal.",
            0.65
        ),
        (
            "Q4 2023 revenue breakdown: Cloud Services $2.1B (+25%), "
            "Enterprise Software $1.5B (+10%), Professional Services $0.6B (+5%).",
            0.95
        ),
    ]
    
    # Example query
    query = "What was the company's revenue in Q4 2023 and what drove the growth?"
    
    print(f"\n[2] Query: {query}")
    print("-" * 60)
    
    print(f"\n[3] Scoring {len(example_documents)} documents...")
    
    # Score documents
    results = agent.score_documents(
        query=query,
        documents=example_documents,
        threshold=0.7
    )
    
    print(f"\n[4] Results ({len(results)} documents above threshold):")
    print("-" * 60)
    
    for i, (content, score, reasoning) in enumerate(results, start=1):
        print(f"\n#{i} - Relevance Score: {score:.2f}")
        print(f"Reasoning: {reasoning}")
        print(f"Content: {content[:150]}...")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")

