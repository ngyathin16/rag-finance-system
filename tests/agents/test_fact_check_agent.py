"""
Tests for Fact Check Agent Module.

Covers:
- FactCheckResult: Pydantic model validation
- FactCheckAgent: verification, hallucination detection, correction triggering
- Edge cases and error handling

All tests use pytest-mock to mock LLM responses for deterministic testing.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from pydantic import ValidationError

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures for common test setup
# =============================================================================

@pytest.fixture
def mock_fact_check_agent_deps():
    """
    Mock all external dependencies for FactCheckAgent.
    
    This fixture mocks:
    - ChatOpenAI (LLM client)
    - ChatPromptTemplate (prompt builder)
    
    Returns a dict with access to all mocks for test assertions.
    """
    with patch("src.agents.fact_check_agent.ChatOpenAI") as mock_chat:
        with patch("src.agents.fact_check_agent.ChatPromptTemplate") as mock_prompt:
            mock_llm = Mock()
            mock_chain = Mock()
            
            # Create a mock prompt that supports | operator for LangChain
            mock_prompt_instance = MagicMock()
            mock_prompt_instance.__or__ = Mock(return_value=mock_chain)
            mock_prompt.from_messages.return_value = mock_prompt_instance
            
            mock_llm.with_structured_output.return_value = mock_chain
            mock_chat.return_value = mock_llm
            
            yield {
                "chat": mock_chat,
                "prompt": mock_prompt,
                "llm": mock_llm,
                "chain": mock_chain
            }


@pytest.fixture
def sample_sources():
    """
    Sample source documents for testing fact verification.
    
    These represent scored documents from the relevance agent
    that were used to generate an answer.
    
    Format: List of (content, relevance_score, reasoning) tuples
    """
    return [
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
            "investments in R&D. Net income margin improved to 22% from 19%.",
            0.78,
            "Related financial metrics for context"
        ),
    ]


@pytest.fixture
def verified_answer():
    """Sample answer that is fully supported by sources."""
    return (
        "The company reported Q4 2023 revenue of $4.2 billion, a 15% increase "
        "year-over-year [1]. Cloud services led the growth at $2.1B (+25%) [2]."
    )


@pytest.fixture
def hallucinated_answer():
    """Sample answer containing hallucinated information."""
    return (
        "The company reported Q4 2023 revenue of $4.2 billion. The CEO announced "
        "plans to acquire a competitor for $2 billion in Q1 2024. Cloud services "
        "drove most of the growth."
    )


@pytest.fixture
def misquoted_answer():
    """Sample answer with incorrect numbers (misquoted from sources)."""
    return (
        "The company reported Q4 2023 revenue of $4.5 billion, a 20% increase "
        "year-over-year. Cloud services revenue was $2.5B with 30% growth."
    )


# =============================================================================
# FactCheckResult Pydantic Model Tests
# =============================================================================

class TestFactCheckResult:
    """Tests for FactCheckResult Pydantic model validation."""
    
    def test_valid_verified_result(self):
        """
        Test creating a valid VERIFIED FactCheckResult.
        
        Validates that the model accepts properly formatted verification
        results when all claims are supported by sources.
        """
        from src.agents.fact_check_agent import FactCheckResult
        
        result = FactCheckResult(
            status="VERIFIED",
            unsupported_claims=[],
            correction_needed=False,
            explanation="All claims match source documents exactly.",
            confidence=0.95
        )
        
        assert result.status == "VERIFIED"
        assert result.unsupported_claims == []
        assert result.correction_needed is False
        assert result.confidence == 0.95
    
    def test_valid_uncertain_result(self):
        """
        Test creating a valid UNCERTAIN FactCheckResult.
        
        Validates model for cases where some claims cannot be verified
        but nothing contradicts the sources.
        """
        from src.agents.fact_check_agent import FactCheckResult
        
        result = FactCheckResult(
            status="UNCERTAIN",
            unsupported_claims=["Competitor acquisition plan"],
            correction_needed=True,
            explanation="Acquisition claim not found in any source document.",
            confidence=0.6
        )
        
        assert result.status == "UNCERTAIN"
        assert len(result.unsupported_claims) == 1
        assert result.correction_needed is True
    
    def test_valid_false_result(self):
        """
        Test creating a valid FALSE FactCheckResult.
        
        Validates model for cases where claims directly contradict sources.
        """
        from src.agents.fact_check_agent import FactCheckResult
        
        result = FactCheckResult(
            status="FALSE",
            unsupported_claims=["Revenue was $4.5B (sources say $4.2B)"],
            correction_needed=True,
            explanation="Revenue figure contradicts source: $4.5B vs $4.2B.",
            confidence=0.92
        )
        
        assert result.status == "FALSE"
        assert result.correction_needed is True
    
    def test_invalid_status_raises_error(self):
        """
        Test that invalid status values raise ValidationError.
        
        Status must be one of: VERIFIED, UNCERTAIN, FALSE.
        """
        from src.agents.fact_check_agent import FactCheckResult
        
        with pytest.raises(ValidationError):
            FactCheckResult(
                status="INVALID",
                unsupported_claims=[],
                correction_needed=False,
                explanation="Test",
                confidence=0.5
            )
    
    def test_confidence_boundaries(self):
        """
        Test that confidence must be between 0.0 and 1.0.
        
        Validates Pydantic field constraints ge=0.0 and le=1.0.
        """
        from src.agents.fact_check_agent import FactCheckResult
        
        # Valid boundaries
        min_result = FactCheckResult(
            status="UNCERTAIN",
            unsupported_claims=[],
            correction_needed=False,
            explanation="Very uncertain",
            confidence=0.0
        )
        assert min_result.confidence == 0.0
        
        max_result = FactCheckResult(
            status="VERIFIED",
            unsupported_claims=[],
            correction_needed=False,
            explanation="Very certain",
            confidence=1.0
        )
        assert max_result.confidence == 1.0
        
        # Invalid values
        with pytest.raises(ValidationError):
            FactCheckResult(
                status="VERIFIED",
                unsupported_claims=[],
                correction_needed=False,
                explanation="Invalid",
                confidence=-0.1
            )
        
        with pytest.raises(ValidationError):
            FactCheckResult(
                status="VERIFIED",
                unsupported_claims=[],
                correction_needed=False,
                explanation="Invalid",
                confidence=1.5
            )


# =============================================================================
# FactCheckAgent Tests
# =============================================================================

class TestFactCheckAgent:
    """Tests for FactCheckAgent class initialization and configuration."""
    
    def test_initialization(self, mock_fact_check_agent_deps):
        """
        Test that FactCheckAgent initializes correctly with default model.
        
        Validates that the agent creates proper LLM chain with low temperature.
        """
        from src.agents.fact_check_agent import FactCheckAgent
        
        agent = FactCheckAgent(model_name="gpt-4o-mini")
        
        assert agent.model_name == "gpt-4o-mini"
    
    def test_empty_model_name_raises_error(self, mock_fact_check_agent_deps):
        """
        Test that empty model name raises ValueError.
        
        Validates input validation for required model_name parameter.
        """
        from src.agents.fact_check_agent import FactCheckAgent
        
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            FactCheckAgent(model_name="")
        
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            FactCheckAgent(model_name=None)


class TestFactCheckAgentVerify:
    """Tests for FactCheckAgent.verify() method."""
    
    def test_verify_detects_hallucinations(
        self, mock_fact_check_agent_deps, sample_sources, hallucinated_answer
    ):
        """
        Test that the agent detects hallucinated information not in sources.
        
        Validates the agent's ability to identify claims that are completely
        fabricated (e.g., acquisition plans that don't exist in sources).
        This is critical for financial Q&A where accuracy is paramount.
        """
        from src.agents.fact_check_agent import FactCheckAgent, FactCheckResult
        
        agent = FactCheckAgent()
        
        # Mock response detecting hallucination
        mock_fact_check_agent_deps["chain"].invoke = Mock(
            return_value=FactCheckResult(
                status="FALSE",
                unsupported_claims=[
                    "CEO announced plans to acquire a competitor for $2 billion"
                ],
                correction_needed=True,
                explanation="The acquisition claim is not found in any source document. "
                           "This appears to be hallucinated information.",
                confidence=0.88
            )
        )
        agent.chain = mock_fact_check_agent_deps["chain"]
        
        result = agent.verify(answer=hallucinated_answer, sources=sample_sources)
        
        # Verify hallucination was detected
        assert result.status in ["FALSE", "UNCERTAIN"]
        assert len(result.unsupported_claims) > 0
        assert result.correction_needed is True
        assert any("acquisition" in claim.lower() or "acquire" in claim.lower() 
                  for claim in result.unsupported_claims)
    
    def test_verify_accepts_accurate_answer(
        self, mock_fact_check_agent_deps, sample_sources, verified_answer
    ):
        """
        Test that the agent accepts an answer fully supported by sources.
        
        Validates that accurate answers receive VERIFIED status and
        don't trigger unnecessary corrections.
        """
        from src.agents.fact_check_agent import FactCheckAgent, FactCheckResult
        
        agent = FactCheckAgent()
        
        # Mock response for verified answer
        mock_fact_check_agent_deps["chain"].invoke = Mock(
            return_value=FactCheckResult(
                status="VERIFIED",
                unsupported_claims=[],
                correction_needed=False,
                explanation="All claims match source documents. Revenue figure of $4.2B "
                           "and 15% growth match [1]. Cloud services $2.1B (+25%) matches [2].",
                confidence=0.95
            )
        )
        agent.chain = mock_fact_check_agent_deps["chain"]
        
        result = agent.verify(answer=verified_answer, sources=sample_sources)
        
        # Verify answer was accepted
        assert result.status == "VERIFIED"
        assert result.unsupported_claims == []
        assert result.correction_needed is False
        assert result.confidence > 0.8
    
    def test_verify_triggers_correction(
        self, mock_fact_check_agent_deps, sample_sources, misquoted_answer
    ):
        """
        Test that misquoted numbers trigger correction requirement.
        
        Validates that the agent identifies numerical discrepancies
        (e.g., $4.5B vs $4.2B, 20% vs 15%) and flags them for correction.
        Financial accuracy requires exact numbers.
        """
        from src.agents.fact_check_agent import FactCheckAgent, FactCheckResult
        
        agent = FactCheckAgent()
        
        # Mock response detecting misquoted numbers
        mock_fact_check_agent_deps["chain"].invoke = Mock(
            return_value=FactCheckResult(
                status="FALSE",
                unsupported_claims=[
                    "Revenue of $4.5 billion (sources say $4.2B)",
                    "20% increase (sources say 15%)",
                    "Cloud services $2.5B with 30% growth (sources say $2.1B, +25%)"
                ],
                correction_needed=True,
                explanation="Multiple numerical discrepancies detected. Revenue, growth "
                           "rate, and cloud services figures all differ from source documents.",
                confidence=0.92
            )
        )
        agent.chain = mock_fact_check_agent_deps["chain"]
        
        result = agent.verify(answer=misquoted_answer, sources=sample_sources)
        
        # Verify correction is required
        assert result.status == "FALSE"
        assert result.correction_needed is True
        assert len(result.unsupported_claims) >= 1
    
    def test_verify_empty_answer_raises_error(
        self, mock_fact_check_agent_deps, sample_sources
    ):
        """
        Test that empty answer raises ValueError.
        
        Validates input validation - an answer is required for verification.
        """
        from src.agents.fact_check_agent import FactCheckAgent
        
        agent = FactCheckAgent()
        
        with pytest.raises(ValueError, match="answer cannot be empty"):
            agent.verify(answer="", sources=sample_sources)
        
        with pytest.raises(ValueError, match="answer cannot be empty"):
            agent.verify(answer="   ", sources=sample_sources)
    
    def test_verify_empty_sources_raises_error(
        self, mock_fact_check_agent_deps, verified_answer
    ):
        """
        Test that empty sources raises ValueError.
        
        Validates that verification requires at least one source document.
        Cannot verify claims without sources to check against.
        """
        from src.agents.fact_check_agent import FactCheckAgent
        
        agent = FactCheckAgent()
        
        with pytest.raises(ValueError, match="sources cannot be empty"):
            agent.verify(answer=verified_answer, sources=[])


class TestFactCheckAgentCorrectionLogic:
    """Tests for FactCheckAgent correction determination logic."""
    
    def test_determine_correction_false_always_corrects(
        self, mock_fact_check_agent_deps
    ):
        """
        Test that FALSE status always triggers correction.
        
        When factual errors are detected, correction is mandatory
        regardless of confidence level.
        """
        from src.agents.fact_check_agent import FactCheckAgent
        
        agent = FactCheckAgent()
        
        # FALSE status should always require correction
        assert agent._determine_correction_needed("FALSE", 0.95) is True
        assert agent._determine_correction_needed("FALSE", 0.5) is True
        assert agent._determine_correction_needed("FALSE", 0.1) is True
    
    def test_determine_correction_verified_never_corrects(
        self, mock_fact_check_agent_deps
    ):
        """
        Test that VERIFIED status never triggers correction.
        
        When all claims are verified, no correction needed regardless
        of confidence level.
        """
        from src.agents.fact_check_agent import FactCheckAgent
        
        agent = FactCheckAgent()
        
        # VERIFIED status should never require correction
        assert agent._determine_correction_needed("VERIFIED", 0.95) is False
        assert agent._determine_correction_needed("VERIFIED", 0.5) is False
        assert agent._determine_correction_needed("VERIFIED", 0.1) is False
    
    def test_determine_correction_uncertain_depends_on_confidence(
        self, mock_fact_check_agent_deps
    ):
        """
        Test that UNCERTAIN status uses confidence threshold.
        
        UNCERTAIN with low confidence (<0.7) triggers correction,
        but high confidence UNCERTAIN does not.
        """
        from src.agents.fact_check_agent import FactCheckAgent
        
        agent = FactCheckAgent()
        
        # Low confidence UNCERTAIN should correct
        assert agent._determine_correction_needed("UNCERTAIN", 0.5) is True
        assert agent._determine_correction_needed("UNCERTAIN", 0.69) is True
        
        # High confidence UNCERTAIN should NOT correct
        assert agent._determine_correction_needed("UNCERTAIN", 0.7) is False
        assert agent._determine_correction_needed("UNCERTAIN", 0.9) is False


class TestFactCheckAgentEdgeCases:
    """Edge case tests for FactCheckAgent."""
    
    def test_verify_with_single_source(
        self, mock_fact_check_agent_deps, verified_answer
    ):
        """
        Test verification with only one source document.
        
        Validates that the agent handles minimal input gracefully.
        """
        from src.agents.fact_check_agent import FactCheckAgent, FactCheckResult
        
        agent = FactCheckAgent()
        
        single_source = [("Revenue was $4.2B in Q4 2023, +15% YoY.", 0.95, "Direct")]
        
        mock_fact_check_agent_deps["chain"].invoke = Mock(
            return_value=FactCheckResult(
                status="VERIFIED",
                unsupported_claims=[],
                correction_needed=False,
                explanation="Revenue claim verified against source.",
                confidence=0.9
            )
        )
        agent.chain = mock_fact_check_agent_deps["chain"]
        
        result = agent.verify(answer="Revenue was $4.2B in Q4 2023.", sources=single_source)
        
        assert result.status == "VERIFIED"
    
    def test_verify_handles_llm_error(
        self, mock_fact_check_agent_deps, sample_sources, verified_answer
    ):
        """
        Test that LLM errors are properly raised.
        
        Validates error propagation for debugging and monitoring.
        """
        from src.agents.fact_check_agent import FactCheckAgent
        
        agent = FactCheckAgent()
        
        mock_fact_check_agent_deps["chain"].invoke = Mock(
            side_effect=Exception("LLM API error")
        )
        agent.chain = mock_fact_check_agent_deps["chain"]
        
        with pytest.raises(Exception, match="LLM API error"):
            agent.verify(answer=verified_answer, sources=sample_sources)
    
    def test_format_sources_includes_all_fields(self, mock_fact_check_agent_deps):
        """
        Test that _format_sources includes content, score, and reasoning.
        
        Validates that all relevant information is passed to LLM for verification.
        """
        from src.agents.fact_check_agent import FactCheckAgent
        
        agent = FactCheckAgent()
        
        sources = [
            ("First document content.", 0.95, "First reasoning"),
            ("Second document content.", 0.85, "Second reasoning"),
        ]
        
        formatted = agent._format_sources(sources)
        
        assert "[Source 1]" in formatted
        assert "[Source 2]" in formatted
        assert "First document content." in formatted
        assert "Second document content." in formatted
        assert "0.95" in formatted
        assert "0.85" in formatted
        assert "First reasoning" in formatted
        assert "Second reasoning" in formatted
    
    def test_verify_partial_hallucination(
        self, mock_fact_check_agent_deps, sample_sources
    ):
        """
        Test detection of partial hallucinations mixed with accurate claims.
        
        Validates that the agent can identify specific hallucinated claims
        even when other parts of the answer are accurate.
        """
        from src.agents.fact_check_agent import FactCheckAgent, FactCheckResult
        
        agent = FactCheckAgent()
        
        # Answer with some accurate and some hallucinated claims
        mixed_answer = (
            "The company reported Q4 2023 revenue of $4.2 billion, a 15% increase. "
            "The CFO resigned in December 2023 due to personal reasons."
        )
        
        mock_fact_check_agent_deps["chain"].invoke = Mock(
            return_value=FactCheckResult(
                status="UNCERTAIN",
                unsupported_claims=[
                    "CFO resigned in December 2023"
                ],
                correction_needed=True,
                explanation="Revenue figures are verified. However, CFO resignation "
                           "claim is not found in any source document.",
                confidence=0.65
            )
        )
        agent.chain = mock_fact_check_agent_deps["chain"]
        
        result = agent.verify(answer=mixed_answer, sources=sample_sources)
        
        # Should identify the specific hallucinated claim
        assert len(result.unsupported_claims) == 1
        assert "CFO" in result.unsupported_claims[0]
        assert result.correction_needed is True
    
    def test_verify_numerical_precision(
        self, mock_fact_check_agent_deps, sample_sources
    ):
        """
        Test verification catches subtle numerical differences.
        
        Validates that even small numerical discrepancies are caught
        (e.g., $4.20B vs $4.2B is OK, but $4.3B vs $4.2B is not).
        """
        from src.agents.fact_check_agent import FactCheckAgent, FactCheckResult
        
        agent = FactCheckAgent()
        
        # Answer with slightly wrong number
        answer_with_wrong_number = "Revenue was $4.3 billion in Q4 2023."
        
        mock_fact_check_agent_deps["chain"].invoke = Mock(
            return_value=FactCheckResult(
                status="FALSE",
                unsupported_claims=[
                    "Revenue of $4.3 billion (sources state $4.2 billion)"
                ],
                correction_needed=True,
                explanation="Revenue figure is incorrect. Sources indicate $4.2B, "
                           "not $4.3B.",
                confidence=0.95
            )
        )
        agent.chain = mock_fact_check_agent_deps["chain"]
        
        result = agent.verify(answer=answer_with_wrong_number, sources=sample_sources)
        
        assert result.status == "FALSE"
        assert result.correction_needed is True

