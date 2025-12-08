"""
Fact Check Agent for Financial Answer Verification.

This module provides a FactCheckAgent that uses LLMs with structured output
to verify if generated answers are supported by source documents. It detects
hallucinations, misquoted data, and logical inconsistencies in financial Q&A.

Features:
    - LLM-based fact verification with structured Pydantic output
    - Detection of unsupported claims and hallucinations
    - Confidence scoring for verification decisions
    - OpenTelemetry instrumentation for observability
    - Automatic correction recommendation based on verification status

Usage:
    from src.agents.fact_check_agent import FactCheckAgent
    
    agent = FactCheckAgent(model_name="gpt-4o-mini")
    
    sources = [
        ("Revenue grew by 15% in Q4 2023...", 0.95, "Direct answer with data"),
        ("Operating expenses increased by 8%...", 0.87, "Related context"),
    ]
    
    result = agent.verify(
        answer="Revenue grew by 15% in Q4 2023, driven by cloud services.",
        sources=sources
    )
    
    if result.correction_needed:
        print(f"Answer needs correction: {result.unsupported_claims}")
"""

import logging
from typing import List, Literal, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from opentelemetry import trace
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Get tracer for OpenTelemetry spans
tracer = trace.get_tracer(__name__)


class FactCheckResult(BaseModel):
    """
    Structured output model for fact verification results.
    
    This model represents the outcome of verifying a generated answer
    against its source documents. It identifies unsupported claims,
    provides a verification status, and recommends whether the answer
    needs to be regenerated.
    
    Attributes:
        status: Verification verdict - VERIFIED if all claims are supported,
                UNCERTAIN if some claims cannot be confirmed, FALSE if claims
                contradict sources
        unsupported_claims: List of specific claims in the answer that are
                           not supported by the provided source documents
        correction_needed: Whether the answer should be regenerated due to
                          unsupported or false claims
        explanation: Detailed reasoning for the verification verdict,
                    explaining what was checked and why
        confidence: Confidence score between 0.0 and 1.0 for the verification
                   decision itself
    
    Example:
        >>> result = FactCheckResult(
        ...     status="VERIFIED",
        ...     unsupported_claims=[],
        ...     correction_needed=False,
        ...     explanation="All claims match source documents exactly.",
        ...     confidence=0.95
        ... )
        
        >>> result = FactCheckResult(
        ...     status="UNCERTAIN",
        ...     unsupported_claims=["Growth rate of 20%"],
        ...     correction_needed=True,
        ...     explanation="Source states 15% growth, not 20%.",
        ...     confidence=0.6
        ... )
    """
    
    status: Literal["VERIFIED", "UNCERTAIN", "FALSE"] = Field(
        description="Verification verdict: VERIFIED (all claims supported), "
                    "UNCERTAIN (some claims unverifiable), FALSE (claims contradict sources)"
    )
    unsupported_claims: List[str] = Field(
        description="Claims not in sources",
        default_factory=list
    )
    correction_needed: bool = Field(
        description="Whether to regenerate"
    )
    explanation: str = Field(
        description="Reasoning for verdict"
    )
    confidence: float = Field(
        description="Confidence in verification",
        ge=0.0,
        le=1.0
    )


# System prompt for strict financial fact-checking
FACT_CHECK_SYSTEM_PROMPT = """You are a strict fact-checker for financial information.

Task: Verify if generated answer's claims are supported by source documents.

Check for:
1. Hallucinations (info not in sources)
2. Misquoted numbers or dates
3. Incorrect attributions
4. Logical inconsistencies

Be thorough - financial accuracy is critical.

Set correction_needed=True if any claim lacks support.

For each claim in the answer:
- Extract the specific factual assertion
- Search all sources for supporting evidence
- Note any discrepancies in numbers, dates, percentages, or entities
- Flag claims that cannot be verified from the provided sources

When determining your verdict:
- VERIFIED: Every factual claim has direct support in sources
- UNCERTAIN: Some claims cannot be verified (missing info) but nothing contradicts sources
- FALSE: One or more claims directly contradict the source information

Your confidence score should reflect how certain you are in your verification verdict."""


FACT_CHECK_USER_PROMPT = """Answer to verify:
{answer}

Source Documents:
{formatted_sources}

Verify each claim in the answer against the sources. List any unsupported claims and determine if correction is needed."""


class FactCheckAgent:
    """
    Agent for verifying generated answers against source documents.
    
    This agent takes a generated answer and its source documents, then
    verifies that all factual claims in the answer are supported by the
    sources. It detects hallucinations, numerical errors, and logical
    inconsistencies that are common in LLM-generated financial content.
    
    The agent is specifically designed for financial Q&A verification,
    where accuracy of numbers, dates, and attributions is critical.
    
    Attributes:
        model_name: Name of the OpenAI model to use
        llm: The ChatOpenAI language model instance
        chain: The prompt-to-structured-output chain
    
    Example:
        >>> agent = FactCheckAgent(model_name="gpt-4o-mini")
        >>> sources = [
        ...     ("Revenue was $4.2B in Q4 2023, up 15% YoY", 0.95, "Direct answer"),
        ...     ("Cloud services drove 60% of revenue", 0.87, "Supporting context"),
        ... ]
        >>> result = agent.verify(
        ...     answer="Revenue was $4.2B in Q4 2023, representing 15% growth. "
        ...            "Cloud services contributed 60% of total revenue.",
        ...     sources=sources
        ... )
        >>> print(f"Status: {result.status}, Correction needed: {result.correction_needed}")
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the FactCheckAgent.
        
        Args:
            model_name: Name of the OpenAI model to use for fact verification.
                       Defaults to "gpt-4o-mini" for cost-effective checking.
        
        Raises:
            ValueError: If model_name is empty or None
        """
        if not model_name:
            raise ValueError("model_name cannot be empty")
        
        self.model_name = model_name
        
        # Initialize LLM with low temperature for consistent verification
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1  # Low temperature for consistent, precise verification
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", FACT_CHECK_SYSTEM_PROMPT),
            ("user", FACT_CHECK_USER_PROMPT)
        ])
        
        # Build chain with structured output
        self.chain = self.prompt | self.llm.with_structured_output(FactCheckResult)
        
        logger.info(f"FactCheckAgent initialized with model: {model_name}")
    
    def _format_sources(self, sources: List[Tuple[str, float, str]]) -> str:
        """
        Format source documents into a numbered list for verification.
        
        Args:
            sources: List of (content, relevance_score, reasoning) tuples
                    from the relevance/generator agents
        
        Returns:
            Formatted string with numbered sources and relevance scores
        """
        formatted_parts = []
        
        for idx, (content, score, reasoning) in enumerate(sources, start=1):
            formatted_parts.append(
                f"[Source {idx}] (Relevance: {score:.2f})\n"
                f"Content: {content}\n"
                f"Relevance Note: {reasoning}"
            )
        
        return "\n\n".join(formatted_parts)
    
    def _determine_correction_needed(self, status: str, confidence: float) -> bool:
        """
        Determine if the answer needs correction based on status and confidence.
        
        The logic follows these rules:
        - Always correct if status is FALSE (factual errors detected)
        - Correct if status is UNCERTAIN and confidence is low (<0.7)
        - No correction needed if status is VERIFIED
        
        Args:
            status: Verification status ("VERIFIED", "UNCERTAIN", or "FALSE")
            confidence: Confidence score for the verification (0.0 to 1.0)
        
        Returns:
            True if the answer should be regenerated, False otherwise
        """
        if status == "FALSE":
            return True
        elif status == "UNCERTAIN" and confidence < 0.7:
            return True
        elif status == "VERIFIED":
            return False
        # Default case for UNCERTAIN with high confidence
        return False
    
    def verify(
        self,
        answer: str,
        sources: List[Tuple[str, float, str]]
    ) -> FactCheckResult:
        """
        Verify if a generated answer's claims are supported by sources.
        
        This method analyzes the answer for factual claims and checks each
        claim against the provided source documents. It identifies unsupported
        claims, hallucinations, and numerical discrepancies.
        
        Args:
            answer: The generated answer text to verify. Should contain
                   factual claims that reference information from sources.
            sources: List of tuples containing 
                    (document_content, relevance_score, reasoning) from
                    the relevance agent or generator.
        
        Returns:
            FactCheckResult containing:
                - status: "VERIFIED", "UNCERTAIN", or "FALSE"
                - unsupported_claims: List of claims not found in sources
                - correction_needed: Whether to regenerate the answer
                - explanation: Detailed reasoning for the verdict
                - confidence: Confidence in the verification decision
        
        Raises:
            ValueError: If answer is empty or sources is empty
        
        Example:
            >>> result = agent.verify(
            ...     answer="Revenue grew 15% to $4.2B in Q4 2023.",
            ...     sources=[("Q4 2023 revenue: $4.2B, +15% YoY", 0.95, "Direct")]
            ... )
            >>> if result.correction_needed:
            ...     print("Need to regenerate answer")
            >>> print(f"Unsupported claims: {result.unsupported_claims}")
        """
        if not answer or not answer.strip():
            raise ValueError("answer cannot be empty")
        
        if not sources:
            raise ValueError("sources cannot be empty")
        
        # Start OpenTelemetry span
        with tracer.start_as_current_span("fact_check_agent.verify") as span:
            # Set input attributes
            span.set_attribute("num_sources", len(sources))
            span.set_attribute("answer_length", len(answer))
            
            # Format sources for the prompt
            formatted_sources = self._format_sources(sources)
            
            try:
                # Invoke the chain to get structured verification result
                result: FactCheckResult = self.chain.invoke({
                    "answer": answer,
                    "formatted_sources": formatted_sources
                })
                
                # Apply correction_needed logic to ensure consistency
                # The LLM provides its assessment, but we enforce our rules
                expected_correction = self._determine_correction_needed(
                    result.status, 
                    result.confidence
                )
                
                # Create final result with enforced correction_needed logic
                final_result = FactCheckResult(
                    status=result.status,
                    unsupported_claims=result.unsupported_claims,
                    correction_needed=expected_correction,
                    explanation=result.explanation,
                    confidence=result.confidence
                )
                
                # Set span attributes for observability
                span.set_attribute("verification_status", final_result.status)
                span.set_attribute("correction_needed", final_result.correction_needed)
                span.set_attribute("num_unsupported_claims", len(final_result.unsupported_claims))
                span.set_attribute("confidence", final_result.confidence)
                
                logger.info(
                    f"Fact check complete: status={final_result.status}, "
                    f"correction_needed={final_result.correction_needed}, "
                    f"unsupported_claims={len(final_result.unsupported_claims)}, "
                    f"confidence={final_result.confidence:.2f}"
                )
                
                return final_result
                
            except Exception as e:
                logger.error(f"Failed to verify answer: {e}")
                span.set_attribute("error", str(e))
                span.set_attribute("error_type", type(e).__name__)
                raise


if __name__ == "__main__":
    """
    Example usage demonstrating the FactCheckAgent.
    
    This example shows how to:
    1. Initialize the FactCheckAgent
    2. Verify an answer against source documents
    3. Handle verification results and corrections
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
    print("Fact Check Agent Demo")
    print("=" * 70)
    
    # Initialize the agent
    print("\n[1] Initializing FactCheckAgent...")
    agent = FactCheckAgent(model_name="gpt-4o-mini")
    print(f"    Model: {agent.model_name}")
    
    # Example sources (from relevance/generator agents)
    example_sources = [
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
    
    # =========================================================================
    # Test Case 1: Verified Answer (all claims supported)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Test Case 1] Verified Answer - All claims supported")
    print("=" * 70)
    
    verified_answer = (
        "The company reported Q4 2023 revenue of $4.2 billion, a 15% increase "
        "year-over-year [1]. Cloud services led the growth at $2.1B (+25%), "
        "followed by Enterprise Software at $1.5B (+10%) [2]."
    )
    
    print(f"\nAnswer to verify:\n{verified_answer}")
    print("-" * 70)
    
    result = agent.verify(answer=verified_answer, sources=example_sources)
    
    print(f"\nVerification Result:")
    print(f"  Status: {result.status}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Correction Needed: {result.correction_needed}")
    print(f"  Unsupported Claims: {result.unsupported_claims}")
    print(f"  Explanation: {result.explanation}")
    
    # =========================================================================
    # Test Case 2: Answer with Hallucination
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Test Case 2] Answer with Hallucination")
    print("=" * 70)
    
    hallucinated_answer = (
        "The company reported Q4 2023 revenue of $4.2 billion, a 15% increase "
        "year-over-year. The CEO announced plans to acquire a major competitor "
        "for $2 billion in Q1 2024. Cloud services drove most of the growth."
    )
    
    print(f"\nAnswer to verify:\n{hallucinated_answer}")
    print("-" * 70)
    
    result = agent.verify(answer=hallucinated_answer, sources=example_sources)
    
    print(f"\nVerification Result:")
    print(f"  Status: {result.status}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Correction Needed: {result.correction_needed}")
    print(f"  Unsupported Claims: {result.unsupported_claims}")
    print(f"  Explanation: {result.explanation}")
    
    # =========================================================================
    # Test Case 3: Answer with Misquoted Numbers
    # =========================================================================
    print("\n" + "=" * 70)
    print("[Test Case 3] Answer with Misquoted Numbers")
    print("=" * 70)
    
    misquoted_answer = (
        "The company reported Q4 2023 revenue of $4.5 billion, a 20% increase "
        "year-over-year. Cloud services revenue was $2.5B with 30% growth."
    )
    
    print(f"\nAnswer to verify:\n{misquoted_answer}")
    print("-" * 70)
    
    result = agent.verify(answer=misquoted_answer, sources=example_sources)
    
    print(f"\nVerification Result:")
    print(f"  Status: {result.status}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Correction Needed: {result.correction_needed}")
    print(f"  Unsupported Claims: {result.unsupported_claims}")
    print(f"  Explanation: {result.explanation}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)

