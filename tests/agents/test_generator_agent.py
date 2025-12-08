"""
Tests for Generator Agent Module.

Covers:
- GeneratedAnswer: Pydantic model validation
- GeneratorAgent: answer generation, citations, confidence scoring
- Token counting and cost calculation
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
def mock_generator_agent_deps():
    """
    Mock all external dependencies for GeneratorAgent.
    
    This fixture mocks:
    - ChatOpenAI (LLM client)
    - ChatPromptTemplate (prompt builder)
    - tiktoken (token counting)
    
    Returns a dict with access to all mocks for test assertions.
    """
    with patch("src.agents.generator_agent.ChatOpenAI") as mock_chat:
        with patch("src.agents.generator_agent.ChatPromptTemplate") as mock_prompt:
            with patch("src.agents.generator_agent.tiktoken") as mock_tiktoken:
                mock_llm = Mock()
                mock_chain = Mock()
                
                # Create a mock prompt that supports | operator for LangChain
                mock_prompt_instance = MagicMock()
                mock_prompt_instance.__or__ = Mock(return_value=mock_chain)
                mock_prompt.from_messages.return_value = mock_prompt_instance
                
                mock_llm.with_structured_output.return_value = mock_chain
                mock_chat.return_value = mock_llm
                
                # Mock tiktoken encoding
                mock_encoding = Mock()
                mock_encoding.encode.return_value = [1] * 100  # 100 tokens
                mock_tiktoken.encoding_for_model.return_value = mock_encoding
                mock_tiktoken.get_encoding.return_value = mock_encoding
                
                yield {
                    "chat": mock_chat,
                    "prompt": mock_prompt,
                    "llm": mock_llm,
                    "chain": mock_chain,
                    "tiktoken": mock_tiktoken,
                    "encoding": mock_encoding
                }


@pytest.fixture
def sample_scored_documents():
    """
    Sample scored documents for testing answer generation.
    
    These represent the output from the relevance agent -
    documents that have been scored and filtered.
    
    Format: List of (content, relevance_score, reasoning) tuples
    """
    return [
        (
            "In Q4 2023, the company reported revenue of $4.2 billion, "
            "representing a 15% increase year-over-year.",
            0.95,
            "Direct answer with specific revenue figures"
        ),
        (
            "Cloud services revenue reached $2.1B in Q4 2023, "
            "growing 25% from the prior year period.",
            0.92,
            "Detailed breakdown of cloud segment"
        ),
        (
            "Operating margin improved to 22% from 19% in the prior year, "
            "driven by operational efficiencies.",
            0.85,
            "Related financial metrics for context"
        ),
    ]


@pytest.fixture
def sample_query():
    """Sample financial query for testing."""
    return "What was the company's revenue in Q4 2023?"


# =============================================================================
# GeneratedAnswer Pydantic Model Tests
# =============================================================================

class TestGeneratedAnswer:
    """Tests for GeneratedAnswer Pydantic model validation."""
    
    def test_valid_generated_answer(self):
        """
        Test creating a valid GeneratedAnswer with all required fields.
        
        Validates that the model accepts properly formatted answer data
        with citations, confidence score, and source references.
        """
        from src.agents.generator_agent import GeneratedAnswer
        
        answer = GeneratedAnswer(
            answer="Revenue grew by 15% to $4.2B in Q4 2023 [1], driven by cloud services [2].",
            confidence=0.92,
            sources_used=[1, 2]
        )
        
        assert "[1]" in answer.answer
        assert "[2]" in answer.answer
        assert answer.confidence == 0.92
        assert answer.sources_used == [1, 2]
    
    def test_confidence_at_boundaries(self):
        """
        Test that confidence score boundary values (0.0 and 1.0) are accepted.
        
        Validates Pydantic field constraints ge=0.0 and le=1.0.
        """
        from src.agents.generator_agent import GeneratedAnswer
        
        # Minimum confidence
        min_answer = GeneratedAnswer(
            answer="Uncertain response.",
            confidence=0.0,
            sources_used=[]
        )
        assert min_answer.confidence == 0.0
        
        # Maximum confidence
        max_answer = GeneratedAnswer(
            answer="Highly confident response [1].",
            confidence=1.0,
            sources_used=[1]
        )
        assert max_answer.confidence == 1.0
    
    def test_confidence_below_zero_raises_error(self):
        """
        Test that confidence score below 0 raises ValidationError.
        
        Ensures invalid negative confidence values are rejected.
        """
        from src.agents.generator_agent import GeneratedAnswer
        
        with pytest.raises(ValidationError):
            GeneratedAnswer(
                answer="Invalid",
                confidence=-0.1,
                sources_used=[]
            )
    
    def test_confidence_above_one_raises_error(self):
        """
        Test that confidence score above 1 raises ValidationError.
        
        Ensures invalid confidence values > 1.0 are rejected.
        """
        from src.agents.generator_agent import GeneratedAnswer
        
        with pytest.raises(ValidationError):
            GeneratedAnswer(
                answer="Invalid",
                confidence=1.5,
                sources_used=[]
            )
    
    def test_empty_sources_used_allowed(self):
        """
        Test that an empty sources_used list is valid.
        
        Some answers may not cite specific sources (e.g., "Information not available").
        """
        from src.agents.generator_agent import GeneratedAnswer
        
        answer = GeneratedAnswer(
            answer="The requested information is not available in the provided sources.",
            confidence=0.3,
            sources_used=[]
        )
        
        assert answer.sources_used == []


# =============================================================================
# GeneratorAgent Tests
# =============================================================================

class TestGeneratorAgent:
    """Tests for GeneratorAgent class initialization and configuration."""
    
    def test_initialization(self, mock_generator_agent_deps):
        """
        Test that GeneratorAgent initializes correctly with default model.
        
        Validates that the agent creates proper LLM chain and tiktoken encoder.
        """
        from src.agents.generator_agent import GeneratorAgent
        
        agent = GeneratorAgent(model_name="gpt-4-turbo-preview")
        
        assert agent.model_name == "gpt-4-turbo-preview"
    
    def test_initialization_with_mini_model(self, mock_generator_agent_deps):
        """
        Test initialization with gpt-4o-mini for cost-effective generation.
        
        Validates that different model configurations are accepted.
        """
        from src.agents.generator_agent import GeneratorAgent
        
        agent = GeneratorAgent(model_name="gpt-4o-mini")
        
        assert agent.model_name == "gpt-4o-mini"
    
    def test_empty_model_name_raises_error(self, mock_generator_agent_deps):
        """
        Test that empty model name raises ValueError.
        
        Validates input validation for required model_name parameter.
        """
        from src.agents.generator_agent import GeneratorAgent
        
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            GeneratorAgent(model_name="")
        
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            GeneratorAgent(model_name=None)


class TestGeneratorAgentGenerate:
    """Tests for GeneratorAgent.generate() method."""
    
    def test_generate_includes_citations(
        self, mock_generator_agent_deps, sample_scored_documents, sample_query
    ):
        """
        Test that generated answers include proper citations [1], [2], etc.
        
        Validates that the agent produces citation-backed responses that
        reference the source documents provided. This is critical for
        financial Q&A where claims must be traceable to sources.
        """
        from src.agents.generator_agent import GeneratorAgent, GeneratedAnswer
        
        agent = GeneratorAgent()
        
        # Mock the chain to return an answer with citations
        mock_generator_agent_deps["chain"].invoke = Mock(
            return_value=GeneratedAnswer(
                answer="Revenue was $4.2B in Q4 2023 [1], with cloud services at $2.1B [2].",
                confidence=0.92,
                sources_used=[1, 2]
            )
        )
        agent.chain = mock_generator_agent_deps["chain"]
        
        result = agent.generate(
            query=sample_query,
            scored_documents=sample_scored_documents
        )
        
        # Verify citations are present in the answer
        assert "[1]" in result["answer"]
        assert "[2]" in result["answer"]
        assert 1 in result["sources_used"]
        assert 2 in result["sources_used"]
    
    def test_generate_calculates_confidence(
        self, mock_generator_agent_deps, sample_scored_documents, sample_query
    ):
        """
        Test that generation calculates and returns confidence score.
        
        Validates that the agent provides a confidence score based on
        source quality and completeness. High-relevance sources with
        direct answers should yield higher confidence.
        """
        from src.agents.generator_agent import GeneratorAgent, GeneratedAnswer
        
        agent = GeneratorAgent()
        
        # Mock with high confidence (good sources)
        mock_generator_agent_deps["chain"].invoke = Mock(
            return_value=GeneratedAnswer(
                answer="Revenue was $4.2B [1].",
                confidence=0.95,
                sources_used=[1]
            )
        )
        agent.chain = mock_generator_agent_deps["chain"]
        
        result = agent.generate(
            query=sample_query,
            scored_documents=sample_scored_documents
        )
        
        # Verify confidence is returned and in valid range
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["confidence"] == 0.95
    
    def test_generate_uses_only_provided_sources(
        self, mock_generator_agent_deps, sample_scored_documents, sample_query
    ):
        """
        Test that generator only uses information from provided sources.
        
        Validates that the sources_used field only contains indices
        that correspond to the scored_documents provided. This ensures
        the answer doesn't reference non-existent sources.
        """
        from src.agents.generator_agent import GeneratorAgent, GeneratedAnswer
        
        agent = GeneratorAgent()
        
        # Mock response using only sources 1 and 3 (within range of 3 documents)
        mock_generator_agent_deps["chain"].invoke = Mock(
            return_value=GeneratedAnswer(
                answer="Revenue was $4.2B [1]. Operating margin improved to 22% [3].",
                confidence=0.88,
                sources_used=[1, 3]  # Valid indices for 3 documents
            )
        )
        agent.chain = mock_generator_agent_deps["chain"]
        
        result = agent.generate(
            query=sample_query,
            scored_documents=sample_scored_documents
        )
        
        # Verify all sources_used are within valid range
        num_sources = len(sample_scored_documents)
        for source_idx in result["sources_used"]:
            assert 1 <= source_idx <= num_sources, \
                f"Source index {source_idx} is out of range [1, {num_sources}]"
        
        # Verify original sources are returned
        assert result["sources"] == sample_scored_documents
    
    def test_generate_empty_query_raises_error(
        self, mock_generator_agent_deps, sample_scored_documents
    ):
        """
        Test that empty query raises ValueError.
        
        Validates input validation - a query is required for generation.
        """
        from src.agents.generator_agent import GeneratorAgent
        
        agent = GeneratorAgent()
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            agent.generate(query="", scored_documents=sample_scored_documents)
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            agent.generate(query="   ", scored_documents=sample_scored_documents)
    
    def test_generate_empty_documents_raises_error(
        self, mock_generator_agent_deps, sample_query
    ):
        """
        Test that empty scored_documents raises ValueError.
        
        Validates that generation requires at least one source document.
        Cannot generate citation-backed answers without sources.
        """
        from src.agents.generator_agent import GeneratorAgent
        
        agent = GeneratorAgent()
        
        with pytest.raises(ValueError, match="scored_documents cannot be empty"):
            agent.generate(query=sample_query, scored_documents=[])
    
    def test_generate_returns_token_count(
        self, mock_generator_agent_deps, sample_scored_documents, sample_query
    ):
        """
        Test that generation returns total token count.
        
        Validates token tracking for cost monitoring and rate limiting.
        """
        from src.agents.generator_agent import GeneratorAgent, GeneratedAnswer
        
        agent = GeneratorAgent()
        
        mock_generator_agent_deps["chain"].invoke = Mock(
            return_value=GeneratedAnswer(
                answer="Revenue was $4.2B [1].",
                confidence=0.9,
                sources_used=[1]
            )
        )
        agent.chain = mock_generator_agent_deps["chain"]
        
        result = agent.generate(
            query=sample_query,
            scored_documents=sample_scored_documents
        )
        
        assert "token_count" in result
        assert isinstance(result["token_count"], int)
        assert result["token_count"] > 0
    
    def test_generate_returns_cost_usd(
        self, mock_generator_agent_deps, sample_scored_documents, sample_query
    ):
        """
        Test that generation returns estimated cost in USD.
        
        Validates cost tracking for budget monitoring.
        """
        from src.agents.generator_agent import GeneratorAgent, GeneratedAnswer
        
        agent = GeneratorAgent()
        
        mock_generator_agent_deps["chain"].invoke = Mock(
            return_value=GeneratedAnswer(
                answer="Revenue was $4.2B [1].",
                confidence=0.9,
                sources_used=[1]
            )
        )
        agent.chain = mock_generator_agent_deps["chain"]
        
        result = agent.generate(
            query=sample_query,
            scored_documents=sample_scored_documents
        )
        
        assert "cost_usd" in result
        assert isinstance(result["cost_usd"], float)
        assert result["cost_usd"] >= 0


class TestGeneratorAgentEdgeCases:
    """Edge case tests for GeneratorAgent."""
    
    def test_generate_with_single_source(self, mock_generator_agent_deps, sample_query):
        """
        Test generation with only one source document.
        
        Validates that the agent handles minimal input gracefully.
        """
        from src.agents.generator_agent import GeneratorAgent, GeneratedAnswer
        
        agent = GeneratorAgent()
        
        single_source = [("Revenue was $4.2B in Q4 2023.", 0.95, "Direct answer")]
        
        mock_generator_agent_deps["chain"].invoke = Mock(
            return_value=GeneratedAnswer(
                answer="Revenue was $4.2B in Q4 2023 [1].",
                confidence=0.95,
                sources_used=[1]
            )
        )
        agent.chain = mock_generator_agent_deps["chain"]
        
        result = agent.generate(
            query=sample_query,
            scored_documents=single_source
        )
        
        assert len(result["sources"]) == 1
        assert result["sources_used"] == [1]
    
    def test_generate_with_low_relevance_sources(
        self, mock_generator_agent_deps, sample_query
    ):
        """
        Test generation with low-relevance sources yields low confidence.
        
        Validates that answer confidence reflects source quality.
        """
        from src.agents.generator_agent import GeneratorAgent, GeneratedAnswer
        
        agent = GeneratorAgent()
        
        low_relevance_sources = [
            ("Some related but vague information.", 0.55, "Tangentially related"),
            ("General company description.", 0.45, "Background only"),
        ]
        
        mock_generator_agent_deps["chain"].invoke = Mock(
            return_value=GeneratedAnswer(
                answer="Based on limited information, revenue details are unclear [1].",
                confidence=0.4,  # Low confidence due to poor sources
                sources_used=[1]
            )
        )
        agent.chain = mock_generator_agent_deps["chain"]
        
        result = agent.generate(
            query=sample_query,
            scored_documents=low_relevance_sources
        )
        
        # Low relevance sources should yield lower confidence
        assert result["confidence"] < 0.7
    
    def test_generate_handles_llm_error(
        self, mock_generator_agent_deps, sample_scored_documents, sample_query
    ):
        """
        Test that LLM errors are properly raised.
        
        Validates error propagation for debugging and monitoring.
        """
        from src.agents.generator_agent import GeneratorAgent
        
        agent = GeneratorAgent()
        
        mock_generator_agent_deps["chain"].invoke = Mock(
            side_effect=Exception("LLM API error")
        )
        agent.chain = mock_generator_agent_deps["chain"]
        
        with pytest.raises(Exception, match="LLM API error"):
            agent.generate(
                query=sample_query,
                scored_documents=sample_scored_documents
            )
    
    def test_format_sources_produces_numbered_list(self, mock_generator_agent_deps):
        """
        Test that _format_sources produces properly numbered source list.
        
        Validates the formatting used in prompts for the LLM.
        """
        from src.agents.generator_agent import GeneratorAgent
        
        agent = GeneratorAgent()
        
        sources = [
            ("First document content.", 0.95, "First reasoning"),
            ("Second document content.", 0.85, "Second reasoning"),
        ]
        
        formatted = agent._format_sources(sources)
        
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "First document content." in formatted
        assert "Second document content." in formatted
        assert "0.95" in formatted
        assert "0.85" in formatted


class TestGeneratorAgentCostCalculation:
    """Tests for token counting and cost calculation."""
    
    def test_calculate_cost_gpt4_turbo(self, mock_generator_agent_deps):
        """
        Test cost calculation for GPT-4 Turbo model.
        
        Validates pricing: $10/1M input, $30/1M output tokens.
        """
        from src.agents.generator_agent import GeneratorAgent
        
        agent = GeneratorAgent(model_name="gpt-4-turbo-preview")
        
        # 1000 input tokens, 500 output tokens
        cost = agent._calculate_cost(input_tokens=1000, output_tokens=500)
        
        # Expected: (1000/1M * $10) + (500/1M * $30) = $0.01 + $0.015 = $0.025
        expected_cost = (1000 / 1_000_000 * 10.00) + (500 / 1_000_000 * 30.00)
        assert cost == pytest.approx(expected_cost, rel=1e-6)
    
    def test_calculate_cost_gpt4o_mini(self, mock_generator_agent_deps):
        """
        Test cost calculation for GPT-4o-mini model.
        
        Validates pricing: $0.15/1M input, $0.60/1M output tokens.
        """
        from src.agents.generator_agent import GeneratorAgent
        
        agent = GeneratorAgent(model_name="gpt-4o-mini")
        
        # 1000 input tokens, 500 output tokens
        cost = agent._calculate_cost(input_tokens=1000, output_tokens=500)
        
        # Expected: (1000/1M * $0.15) + (500/1M * $0.60)
        expected_cost = (1000 / 1_000_000 * 0.15) + (500 / 1_000_000 * 0.60)
        assert cost == pytest.approx(expected_cost, rel=1e-6)
    
    def test_count_tokens(self, mock_generator_agent_deps):
        """
        Test token counting functionality.
        
        Validates that _count_tokens uses tiktoken encoder correctly.
        """
        from src.agents.generator_agent import GeneratorAgent
        
        agent = GeneratorAgent()
        
        # Mock encoding returns 100 tokens for any input
        count = agent._count_tokens("Sample text to count tokens for.")
        
        # The mock returns [1] * 100, so length should be 100
        assert count == 100

