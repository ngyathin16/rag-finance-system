"""
Generator Agent for Financial Answer Generation.

This module provides a GeneratorAgent that uses LLMs with structured output
to generate precise, citation-backed answers from scored financial documents.

Features:
    - LLM-based answer generation with inline citations
    - Structured Pydantic output with confidence scoring
    - OpenTelemetry instrumentation for observability
    - Token counting and cost tracking via tiktoken
    - Source quality and completeness assessment

Usage:
    from src.agents.generator_agent import GeneratorAgent
    
    agent = GeneratorAgent(model_name="gpt-4-turbo-preview")
    
    scored_documents = [
        ("Revenue grew by 15% in Q4...", 0.95, "Direct answer with data"),
        ("Operating expenses increased...", 0.87, "Related financial context"),
    ]
    
    result = agent.generate(
        query="What was the revenue growth?",
        scored_documents=scored_documents
    )
"""

import logging
from typing import List, Tuple

import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from opentelemetry import trace
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Get tracer for OpenTelemetry spans
tracer = trace.get_tracer(__name__)

# Pricing per 1M tokens (USD) for different models
MODEL_PRICING = {
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


class GeneratedAnswer(BaseModel):
    """
    Structured output model for generated financial answers.
    
    This model enforces a consistent format for LLM responses when
    generating answers with citations from source documents.
    
    Attributes:
        answer: The generated answer with inline citations [1][2]
        confidence: Confidence score between 0.0 and 1.0
        sources_used: List of source indices that were cited in the answer
    
    Example:
        >>> answer = GeneratedAnswer(
        ...     answer="Revenue grew by 15% in Q4 2023 [1], driven by cloud services [2].",
        ...     confidence=0.92,
        ...     sources_used=[1, 2]
        ... )
    """
    
    answer: str = Field(
        description="Answer with citations [1][2]"
    )
    confidence: float = Field(
        description="Confidence 0-1",
        ge=0.0,
        le=1.0
    )
    sources_used: List[int] = Field(
        description="Source indices cited"
    )


# System prompt for financial answer generation
GENERATOR_SYSTEM_PROMPT = """You are a financial analyst generating precise answers.

Rules:
1. Use ONLY information from provided sources
2. Include inline citations [1], [2], [3] for each claim
3. For numbers, quote exactly with citation
4. If information is missing, state what's unavailable
5. Estimate confidence based on source quality and completeness

Format your response with clear citations."""

GENERATOR_USER_PROMPT = """Question: {query}

Sources:
{formatted_sources}

Generate a precise answer with citations to the sources above."""


class GeneratorAgent:
    """
    Agent for generating citation-backed answers from scored documents.
    
    This agent takes scored and filtered documents from the relevance agent
    and generates a comprehensive answer with inline citations. It tracks
    token usage and costs, and provides confidence scoring based on
    source quality and completeness.
    
    The agent is specifically designed for financial Q&A, emphasizing
    precise quotation of numbers and clear attribution of claims.
    
    Attributes:
        model_name: Name of the OpenAI model to use
        llm: The ChatOpenAI language model instance
        chain: The prompt-to-structured-output chain
        encoding: Tiktoken encoding for token counting
    
    Example:
        >>> agent = GeneratorAgent(model_name="gpt-4-turbo-preview")
        >>> docs = [
        ...     ("Revenue increased 20% YoY in 2023", 0.95, "Direct answer"),
        ...     ("Cloud services drove the growth", 0.87, "Supporting context"),
        ... ]
        >>> result = agent.generate(
        ...     query="What was the revenue growth?",
        ...     scored_documents=docs
        ... )
        >>> print(result["answer"])
        >>> print(f"Cost: ${result['cost_usd']:.4f}")
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the GeneratorAgent.
        
        Args:
            model_name: Name of the OpenAI model to use for answer generation.
                       Defaults to "gpt-4o" for high-quality generation (70% cheaper, better performance).
        
        Raises:
            ValueError: If model_name is empty or None
        """
        if not model_name:
            raise ValueError("model_name cannot be empty")
        
        self.model_name = model_name
        
        # Initialize LLM with moderate temperature for natural responses
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3  # Moderate temperature for balanced responses
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", GENERATOR_SYSTEM_PROMPT),
            ("user", GENERATOR_USER_PROMPT)
        ])
        
        # Build chain with structured output
        self.chain = self.prompt | self.llm.with_structured_output(GeneratedAnswer)
        
        # Initialize tiktoken encoding for token counting
        try:
            # Try to get encoding for specific model
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fall back to cl100k_base for newer models
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"GeneratorAgent initialized with model: {model_name}")
    
    def _format_sources(self, scored_documents: List[Tuple[str, float, str]]) -> str:
        """
        Format scored documents into a numbered source list.
        
        Args:
            scored_documents: List of (content, relevance_score, reasoning) tuples
        
        Returns:
            Formatted string with numbered sources and relevance scores
        """
        formatted_parts = []
        
        for idx, (content, score, _reasoning) in enumerate(scored_documents, start=1):
            formatted_parts.append(
                f"[{idx}] (Relevance: {score:.2f})\n{content}"
            )
        
        return "\n\n".join(formatted_parts)
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: The text to count tokens for
        
        Returns:
            Number of tokens in the text
        """
        return len(self.encoding.encode(text))
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost in USD for the API call.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Cost in USD
        """
        pricing = MODEL_PRICING.get(
            self.model_name,
            {"input": 10.00, "output": 30.00}  # Default to GPT-4 Turbo pricing
        )
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def generate(
        self,
        query: str,
        scored_documents: List[Tuple[str, float, str]]
    ) -> dict:
        """
        Generate a citation-backed answer from scored documents.
        
        This method takes the query and relevant documents, formats them
        appropriately, and uses the LLM to generate a comprehensive answer
        with inline citations. It tracks token usage and calculates costs.
        
        Args:
            query: The user's financial question
            scored_documents: List of tuples containing 
                (document_content, relevance_score, reasoning) from the
                relevance agent.
        
        Returns:
            Dictionary containing:
                - answer (str): The generated answer with citations
                - confidence (float): Confidence score 0-1
                - sources (List[Tuple]): Original scored_documents
                - sources_used (List[int]): Indices of sources cited
                - token_count (int): Total tokens used
                - cost_usd (float): Estimated cost in USD
        
        Raises:
            ValueError: If query is empty or scored_documents is empty
        
        Example:
            >>> result = agent.generate(
            ...     query="What was Q4 revenue?",
            ...     scored_documents=[("Revenue was $4.2B...", 0.95, "Direct answer")]
            ... )
            >>> print(result["answer"])
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        
        if not scored_documents:
            raise ValueError("scored_documents cannot be empty")
        
        # Start OpenTelemetry span
        with tracer.start_as_current_span("generator_agent.generate") as span:
            # Set input attributes
            span.set_attribute("num_sources_provided", len(scored_documents))
            span.set_attribute("query_length", len(query))
            
            # Format sources for the prompt
            formatted_sources = self._format_sources(scored_documents)
            
            # Build the full prompt for token counting
            full_prompt = (
                f"{GENERATOR_SYSTEM_PROMPT}\n\n"
                f"Question: {query}\n\n"
                f"Sources:\n{formatted_sources}\n\n"
                f"Generate a precise answer with citations to the sources above."
            )
            input_tokens = self._count_tokens(full_prompt)
            
            try:
                # Invoke the chain to get structured answer
                result: GeneratedAnswer = self.chain.invoke({
                    "query": query,
                    "formatted_sources": formatted_sources
                })
                
                # Count output tokens
                output_text = result.answer
                output_tokens = self._count_tokens(output_text)
                total_tokens = input_tokens + output_tokens
                
                # Calculate cost
                cost_usd = self._calculate_cost(input_tokens, output_tokens)
                
                # Set span attributes
                span.set_attribute("num_sources_used", len(result.sources_used))
                span.set_attribute("confidence_score", result.confidence)
                span.set_attribute("answer_length", len(result.answer))
                span.set_attribute("input_tokens", input_tokens)
                span.set_attribute("output_tokens", output_tokens)
                span.set_attribute("total_tokens", total_tokens)
                span.set_attribute("cost_usd", cost_usd)
                
                logger.info(
                    f"Answer generated: confidence={result.confidence:.2f}, "
                    f"sources_used={result.sources_used}, "
                    f"tokens={total_tokens}, cost=${cost_usd:.4f}"
                )
                
                return {
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "sources": scored_documents,
                    "sources_used": result.sources_used,
                    "token_count": total_tokens,
                    "cost_usd": cost_usd
                }
                
            except Exception as e:
                logger.error(f"Failed to generate answer: {e}")
                span.set_attribute("error", str(e))
                span.set_attribute("error_type", type(e).__name__)
                raise


if __name__ == "__main__":
    """
    Example usage demonstrating the GeneratorAgent.
    
    This example shows how to:
    1. Initialize the GeneratorAgent
    2. Generate an answer from scored documents
    3. Display the result with cost information
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
    print("Generator Agent Demo")
    print("=" * 60)
    
    # Initialize the agent
    print("\n[1] Initializing GeneratorAgent...")
    agent = GeneratorAgent(model_name="gpt-4-turbo-preview")
    print(f"    Model: {agent.model_name}")
    
    # Example scored documents (from relevance agent)
    example_scored_documents = [
        (
            "In Q4 2023, the company reported revenue of $4.2 billion, "
            "representing a 15% increase year-over-year. The growth was "
            "primarily driven by strong performance in cloud services.",
            0.95,
            "Direct answer with specific Q4 2023 revenue figures"
        ),
        (
            "Q4 2023 revenue breakdown: Cloud Services $2.1B (+25%), "
            "Enterprise Software $1.5B (+10%), Professional Services $0.6B (+5%).",
            0.92,
            "Detailed revenue breakdown by segment"
        ),
        (
            "Operating expenses increased by 8% in 2023, primarily due to "
            "investments in R&D and sales expansion. Net income margin "
            "improved to 22% from 19% in the prior year.",
            0.78,
            "Related financial metrics for context"
        ),
    ]
    
    # Example query
    query = "What was the company's revenue in Q4 2023 and what drove the growth?"
    
    print(f"\n[2] Query: {query}")
    print("-" * 60)
    
    print(f"\n[3] Generating answer from {len(example_scored_documents)} sources...")
    
    # Generate answer
    result = agent.generate(
        query=query,
        scored_documents=example_scored_documents
    )
    
    print(f"\n[4] Generated Answer:")
    print("-" * 60)
    print(result["answer"])
    
    print(f"\n[5] Metadata:")
    print("-" * 60)
    print(f"    Confidence: {result['confidence']:.2f}")
    print(f"    Sources Used: {result['sources_used']}")
    print(f"    Token Count: {result['token_count']}")
    print(f"    Cost: ${result['cost_usd']:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")

