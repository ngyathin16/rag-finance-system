"""
Integration Tests for SelfCorrectingRAG Orchestrator.

Covers:
- Full pipeline with verification
- Self-correction loop behavior
- Max corrections limit enforcement
- Edge cases and error handling

These tests mock the underlying agents and vector store to test
the orchestration logic in isolation.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Fixtures for common test setup
# =============================================================================

@pytest.fixture
def mock_vector_store():
    """
    Mock vector store for integration testing.
    
    Creates a mock BaseVectorStore with sample financial documents
    that simulates similarity search results.
    
    Returns:
        Mock vector store with pre-configured search results
    """
    mock_store = Mock()
    
    # Sample documents representing financial filing excerpts
    sample_docs = [
        (
            Mock(
                page_content="In Q4 2023, the company reported revenue of $4.2 billion, "
                           "representing a 15% increase year-over-year.",
                metadata={"source": "10k_2023.txt", "section": "Item 7"}
            ),
            0.95  # Similarity score
        ),
        (
            Mock(
                page_content="Cloud services revenue reached $2.1B in Q4 2023, "
                           "growing 25% from the prior year period.",
                metadata={"source": "10k_2023.txt", "section": "Item 7"}
            ),
            0.92
        ),
        (
            Mock(
                page_content="Operating expenses increased by 8% in 2023, primarily due to "
                           "investments in R&D. Net income margin improved to 22% from 19%.",
                metadata={"source": "10k_2023.txt", "section": "Item 7"}
            ),
            0.85
        ),
    ]
    
    mock_store.similarity_search.return_value = sample_docs
    mock_store.get_collection_stats.return_value = {
        "name": "test_collection",
        "count": 100
    }
    
    return mock_store


@pytest.fixture
def sample_scored_documents():
    """
    Sample scored documents (output of relevance agent).
    
    Format: List of (content, relevance_score, reasoning) tuples
    """
    return [
        (
            "In Q4 2023, the company reported revenue of $4.2 billion, "
            "representing a 15% increase year-over-year.",
            0.95,
            "Direct answer with specific Q4 2023 revenue figures"
        ),
        (
            "Cloud services revenue reached $2.1B in Q4 2023, "
            "growing 25% from the prior year period.",
            0.92,
            "Detailed breakdown of cloud segment"
        ),
        (
            "Operating expenses increased by 8% in 2023. "
            "Net income margin improved to 22% from 19%.",
            0.85,
            "Related financial metrics"
        ),
    ]


@pytest.fixture
def mock_orchestrator_agents(mock_vector_store, sample_scored_documents):
    """
    Mock all agents used by SelfCorrectingRAG.
    
    This fixture patches:
    - RelevanceAgent: Document scoring and filtering
    - GeneratorAgent: Answer generation with citations
    - FactCheckAgent: Answer verification
    - Tracing/metrics components
    
    Returns a dict with access to all mocks for test assertions.
    """
    with patch("src.orchestrator.RelevanceAgent") as mock_relevance:
        with patch("src.orchestrator.GeneratorAgent") as mock_generator:
            with patch("src.orchestrator.FactCheckAgent") as mock_fact_check:
                with patch("src.orchestrator.setup_tracing"):
                    with patch("src.orchestrator.get_metrics_collector"):
                        # Configure RelevanceAgent mock
                        mock_relevance_instance = Mock()
                        mock_relevance_instance.score_documents.return_value = sample_scored_documents
                        mock_relevance.return_value = mock_relevance_instance
                        
                        # Configure GeneratorAgent mock
                        mock_generator_instance = Mock()
                        mock_generator.return_value = mock_generator_instance
                        
                        # Configure FactCheckAgent mock
                        mock_fact_check_instance = Mock()
                        mock_fact_check.return_value = mock_fact_check_instance
                        
                        yield {
                            "vector_store": mock_vector_store,
                            "relevance_agent": mock_relevance_instance,
                            "generator_agent": mock_generator_instance,
                            "fact_check_agent": mock_fact_check_instance,
                            "RelevanceAgent": mock_relevance,
                            "GeneratorAgent": mock_generator,
                            "FactCheckAgent": mock_fact_check,
                        }


# =============================================================================
# SelfCorrectingRAG Initialization Tests
# =============================================================================

class TestSelfCorrectingRAGInitialization:
    """Tests for SelfCorrectingRAG initialization and configuration."""
    
    def test_initialization_with_valid_params(self, mock_orchestrator_agents):
        """
        Test that SelfCorrectingRAG initializes correctly with valid parameters.
        
        Validates that all agents are initialized and configuration is stored.
        """
        from src.orchestrator import SelfCorrectingRAG
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        assert rag.max_corrections == 2
        assert rag.vector_store is not None
        assert rag.relevance_agent is not None
        assert rag.generator_agent is not None
        assert rag.fact_check_agent is not None
    
    def test_initialization_none_vector_store_raises_error(self, mock_orchestrator_agents):
        """
        Test that None vector_store raises ValueError.
        
        A vector store is required for document retrieval.
        """
        from src.orchestrator import SelfCorrectingRAG
        
        with pytest.raises(ValueError, match="vector_store cannot be None"):
            SelfCorrectingRAG(vector_store=None, max_corrections=2)
    
    def test_initialization_negative_max_corrections_raises_error(
        self, mock_orchestrator_agents
    ):
        """
        Test that negative max_corrections raises ValueError.
        
        max_corrections must be non-negative (0 means no corrections allowed).
        """
        from src.orchestrator import SelfCorrectingRAG
        
        with pytest.raises(ValueError, match="max_corrections must be non-negative"):
            SelfCorrectingRAG(
                vector_store=mock_orchestrator_agents["vector_store"],
                max_corrections=-1
            )
    
    def test_initialization_zero_corrections_allowed(self, mock_orchestrator_agents):
        """
        Test that max_corrections=0 is valid (no corrections allowed).
        
        Some use cases may want fast responses without self-correction.
        """
        from src.orchestrator import SelfCorrectingRAG
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=0
        )
        
        assert rag.max_corrections == 0


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================

class TestFullPipelineWithVerification:
    """Tests for complete RAG pipeline with fact-checking verification."""
    
    def test_full_pipeline_with_verification(self, mock_orchestrator_agents):
        """
        Test the complete pipeline from query to verified answer.
        
        Validates that:
        1. Documents are retrieved from vector store
        2. Relevance filtering is applied
        3. Answer is generated with citations
        4. Answer is fact-checked and verified
        5. Result contains all expected fields
        
        This is the happy path where the first answer is verified.
        """
        from src.orchestrator import SelfCorrectingRAG
        from src.agents.fact_check_agent import FactCheckResult
        
        # Configure mocks for successful first-attempt verification
        mock_orchestrator_agents["generator_agent"].generate.return_value = {
            "answer": "Revenue was $4.2B in Q4 2023, a 15% increase [1]. "
                     "Cloud services reached $2.1B [2].",
            "confidence": 0.92,
            "sources": mock_orchestrator_agents["relevance_agent"].score_documents.return_value,
            "sources_used": [1, 2],
            "token_count": 500,
            "cost_usd": 0.015
        }
        
        mock_orchestrator_agents["fact_check_agent"].verify.return_value = FactCheckResult(
            status="VERIFIED",
            unsupported_claims=[],
            correction_needed=False,
            explanation="All claims match source documents.",
            confidence=0.95
        )
        
        # Create orchestrator and run query
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        result = rag.query("What was the company's revenue in Q4 2023?")
        
        # Verify complete pipeline executed
        assert mock_orchestrator_agents["vector_store"].similarity_search.called
        assert mock_orchestrator_agents["relevance_agent"].score_documents.called
        assert mock_orchestrator_agents["generator_agent"].generate.called
        assert mock_orchestrator_agents["fact_check_agent"].verify.called
        
        # Verify result structure
        assert "answer" in result
        assert "confidence" in result
        assert "verification_status" in result
        assert "sources" in result
        assert "corrections_made" in result
        assert "latency" in result
        assert "total_tokens" in result
        assert "total_cost_usd" in result
        
        # Verify values
        assert result["verification_status"] == "VERIFIED"
        assert result["corrections_made"] == 0
        assert "[1]" in result["answer"]
        assert "[2]" in result["answer"]
    
    def test_pipeline_returns_metrics(self, mock_orchestrator_agents):
        """
        Test that pipeline returns comprehensive metrics.
        
        Validates token counts, costs, and latency are tracked correctly.
        """
        from src.orchestrator import SelfCorrectingRAG
        from src.agents.fact_check_agent import FactCheckResult
        
        mock_orchestrator_agents["generator_agent"].generate.return_value = {
            "answer": "Revenue was $4.2B [1].",
            "confidence": 0.9,
            "sources": [],
            "sources_used": [1],
            "token_count": 750,
            "cost_usd": 0.025
        }
        
        mock_orchestrator_agents["fact_check_agent"].verify.return_value = FactCheckResult(
            status="VERIFIED",
            unsupported_claims=[],
            correction_needed=False,
            explanation="Verified.",
            confidence=0.95
        )
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        result = rag.query("What was revenue?")
        
        # Verify metrics are present and reasonable
        assert result["total_tokens"] == 750
        assert result["total_cost_usd"] == 0.025
        assert result["latency"] > 0
        assert isinstance(result["latency"], float)


# =============================================================================
# Self-Correction Loop Tests
# =============================================================================

class TestSelfCorrectionLoop:
    """Tests for the self-correction loop behavior."""
    
    def test_self_correction_loop(self, mock_orchestrator_agents):
        """
        Test that correction loop triggers when fact-check fails.
        
        Validates that:
        1. First answer with hallucination triggers correction
        2. Generator is called again with correction context
        3. Second verified answer is accepted
        4. corrections_made counter is incremented
        
        This tests the core self-correction mechanism.
        """
        from src.orchestrator import SelfCorrectingRAG
        from src.agents.fact_check_agent import FactCheckResult
        
        # First attempt: hallucinated answer
        first_answer = {
            "answer": "Revenue was $4.2B [1]. CEO announced acquisition plans.",
            "confidence": 0.85,
            "sources": [],
            "sources_used": [1],
            "token_count": 500,
            "cost_usd": 0.015
        }
        
        # Second attempt: corrected answer
        second_answer = {
            "answer": "Revenue was $4.2B in Q4 2023, a 15% increase [1].",
            "confidence": 0.92,
            "sources": [],
            "sources_used": [1],
            "token_count": 450,
            "cost_usd": 0.013
        }
        
        # Configure generator to return different answers on each call
        mock_orchestrator_agents["generator_agent"].generate.side_effect = [
            first_answer,
            second_answer
        ]
        
        # First verification fails, second succeeds
        mock_orchestrator_agents["fact_check_agent"].verify.side_effect = [
            FactCheckResult(
                status="UNCERTAIN",
                unsupported_claims=["CEO announced acquisition plans"],
                correction_needed=True,
                explanation="Acquisition claim not in sources.",
                confidence=0.6
            ),
            FactCheckResult(
                status="VERIFIED",
                unsupported_claims=[],
                correction_needed=False,
                explanation="All claims verified.",
                confidence=0.95
            )
        ]
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        result = rag.query("What was revenue?")
        
        # Verify correction loop executed
        assert mock_orchestrator_agents["generator_agent"].generate.call_count == 2
        assert mock_orchestrator_agents["fact_check_agent"].verify.call_count == 2
        
        # Verify result reflects corrected answer
        assert result["verification_status"] == "VERIFIED"
        assert result["corrections_made"] == 1
        
        # Verify tokens/cost accumulated from both attempts
        assert result["total_tokens"] == 500 + 450
        assert result["total_cost_usd"] == pytest.approx(0.015 + 0.013, rel=1e-6)
    
    def test_correction_adds_context_to_documents(self, mock_orchestrator_agents):
        """
        Test that correction context is added to documents for retry.
        
        Validates that unsupported claims from fact-checker are passed
        to the generator as correction context to guide the retry.
        """
        from src.orchestrator import SelfCorrectingRAG
        from src.agents.fact_check_agent import FactCheckResult
        
        # First attempt fails
        mock_orchestrator_agents["generator_agent"].generate.side_effect = [
            {
                "answer": "Bad answer with hallucination.",
                "confidence": 0.7,
                "sources": [],
                "sources_used": [1],
                "token_count": 500,
                "cost_usd": 0.015
            },
            {
                "answer": "Corrected answer [1].",
                "confidence": 0.9,
                "sources": [],
                "sources_used": [1],
                "token_count": 400,
                "cost_usd": 0.012
            }
        ]
        
        unsupported_claims = ["The CEO resigned", "Quarterly dividend increased"]
        
        mock_orchestrator_agents["fact_check_agent"].verify.side_effect = [
            FactCheckResult(
                status="FALSE",
                unsupported_claims=unsupported_claims,
                correction_needed=True,
                explanation="Multiple unsupported claims.",
                confidence=0.85
            ),
            FactCheckResult(
                status="VERIFIED",
                unsupported_claims=[],
                correction_needed=False,
                explanation="Verified.",
                confidence=0.95
            )
        ]
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        result = rag.query("What happened?")
        
        # Verify second generate call received correction context
        second_call_args = mock_orchestrator_agents["generator_agent"].generate.call_args_list[1]
        second_docs = second_call_args[1]["scored_documents"]
        
        # First document should be correction context
        correction_doc = second_docs[0]
        assert "CORRECTION CONTEXT" in correction_doc[0]
        for claim in unsupported_claims:
            assert claim in correction_doc[0]


# =============================================================================
# Max Corrections Limit Tests
# =============================================================================

class TestMaxCorrectionsLimit:
    """Tests for max corrections limit enforcement."""
    
    def test_max_corrections_limit(self, mock_orchestrator_agents):
        """
        Test that max_corrections limit is enforced.
        
        Validates that:
        1. Orchestrator stops after max_corrections attempts
        2. Warning is included in result
        3. Best available answer is returned even if not verified
        
        This prevents infinite correction loops.
        """
        from src.orchestrator import SelfCorrectingRAG
        from src.agents.fact_check_agent import FactCheckResult
        
        # All attempts fail verification
        failed_answer = {
            "answer": "Answer with persistent hallucination.",
            "confidence": 0.7,
            "sources": [],
            "sources_used": [1],
            "token_count": 500,
            "cost_usd": 0.015
        }
        
        mock_orchestrator_agents["generator_agent"].generate.return_value = failed_answer
        
        # All verifications trigger correction
        mock_orchestrator_agents["fact_check_agent"].verify.return_value = FactCheckResult(
            status="UNCERTAIN",
            unsupported_claims=["Persistent hallucination"],
            correction_needed=True,
            explanation="Cannot verify claim.",
            confidence=0.5
        )
        
        # Set max_corrections=2, so 3 total attempts (initial + 2 corrections)
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        result = rag.query("What was revenue?")
        
        # Verify max attempts enforced: 1 initial + 2 corrections = 3 total
        assert mock_orchestrator_agents["generator_agent"].generate.call_count == 3
        assert mock_orchestrator_agents["fact_check_agent"].verify.call_count == 3
        
        # Verify result
        assert result["corrections_made"] == 2
        assert result["verification_status"] == "UNCERTAIN"
        assert "warning" in result
        assert "Maximum corrections" in result["warning"]
    
    def test_zero_corrections_returns_first_answer(self, mock_orchestrator_agents):
        """
        Test that max_corrections=0 returns first answer without correction.
        
        Validates fast-path mode where no corrections are attempted,
        even if fact-check would trigger correction.
        """
        from src.orchestrator import SelfCorrectingRAG
        from src.agents.fact_check_agent import FactCheckResult
        
        mock_orchestrator_agents["generator_agent"].generate.return_value = {
            "answer": "First answer [1].",
            "confidence": 0.8,
            "sources": [],
            "sources_used": [1],
            "token_count": 500,
            "cost_usd": 0.015
        }
        
        # Verification would trigger correction, but max_corrections=0
        mock_orchestrator_agents["fact_check_agent"].verify.return_value = FactCheckResult(
            status="UNCERTAIN",
            unsupported_claims=["Some claim"],
            correction_needed=True,
            explanation="Would need correction.",
            confidence=0.6
        )
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=0  # No corrections allowed
        )
        
        result = rag.query("What was revenue?")
        
        # Only one attempt should be made
        assert mock_orchestrator_agents["generator_agent"].generate.call_count == 1
        assert result["corrections_made"] == 0
        
        # Warning should be present since verification failed
        assert "warning" in result
    
    def test_stops_early_when_verified(self, mock_orchestrator_agents):
        """
        Test that loop stops immediately when answer is verified.
        
        Even with high max_corrections, no further attempts if verified.
        """
        from src.orchestrator import SelfCorrectingRAG
        from src.agents.fact_check_agent import FactCheckResult
        
        mock_orchestrator_agents["generator_agent"].generate.return_value = {
            "answer": "Perfect answer [1].",
            "confidence": 0.95,
            "sources": [],
            "sources_used": [1],
            "token_count": 500,
            "cost_usd": 0.015
        }
        
        mock_orchestrator_agents["fact_check_agent"].verify.return_value = FactCheckResult(
            status="VERIFIED",
            unsupported_claims=[],
            correction_needed=False,
            explanation="All verified.",
            confidence=0.98
        )
        
        # Even with high max_corrections
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=10
        )
        
        result = rag.query("What was revenue?")
        
        # Only one attempt since first answer verified
        assert mock_orchestrator_agents["generator_agent"].generate.call_count == 1
        assert result["corrections_made"] == 0
        assert result["verification_status"] == "VERIFIED"
        assert "warning" not in result


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestOrchestratorEdgeCases:
    """Edge case and error handling tests for SelfCorrectingRAG."""
    
    def test_empty_query_raises_error(self, mock_orchestrator_agents):
        """
        Test that empty query raises ValueError.
        
        Validates input validation - a question is required.
        """
        from src.orchestrator import SelfCorrectingRAG
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        with pytest.raises(ValueError, match="question cannot be empty"):
            rag.query("")
        
        with pytest.raises(ValueError, match="question cannot be empty"):
            rag.query("   ")
    
    def test_insufficient_documents_raises_error(self, mock_orchestrator_agents):
        """
        Test that insufficient relevant documents raises InsufficientDocumentsError.
        
        Validates that the orchestrator properly handles cases where
        no documents pass the relevance threshold.
        """
        from src.orchestrator import SelfCorrectingRAG, InsufficientDocumentsError
        
        # Configure relevance agent to return empty list (all filtered out)
        mock_orchestrator_agents["relevance_agent"].score_documents.return_value = []
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        with pytest.raises(InsufficientDocumentsError) as exc_info:
            rag.query("What was revenue?")
        
        # Verify exception contains helpful information
        assert "Insufficient relevant documents" in str(exc_info.value)
    
    def test_retrieval_error_propagates(self, mock_orchestrator_agents):
        """
        Test that vector store errors are properly propagated.
        
        Validates that retrieval failures don't get silently swallowed.
        """
        from src.orchestrator import SelfCorrectingRAG
        
        mock_orchestrator_agents["vector_store"].similarity_search.side_effect = \
            Exception("Vector store connection failed")
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        with pytest.raises(Exception, match="Vector store connection failed"):
            rag.query("What was revenue?")
    
    def test_generation_error_propagates(self, mock_orchestrator_agents):
        """
        Test that generation errors are properly propagated.
        
        Validates that LLM failures during generation don't get swallowed.
        """
        from src.orchestrator import SelfCorrectingRAG
        
        mock_orchestrator_agents["generator_agent"].generate.side_effect = \
            Exception("LLM API error")
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        with pytest.raises(Exception, match="LLM API error"):
            rag.query("What was revenue?")
    
    def test_fact_check_error_propagates(self, mock_orchestrator_agents):
        """
        Test that fact-checking errors are properly propagated.
        
        Validates that verification failures don't get silently swallowed.
        """
        from src.orchestrator import SelfCorrectingRAG
        
        mock_orchestrator_agents["generator_agent"].generate.return_value = {
            "answer": "Some answer.",
            "confidence": 0.8,
            "sources": [],
            "sources_used": [1],
            "token_count": 500,
            "cost_usd": 0.015
        }
        
        mock_orchestrator_agents["fact_check_agent"].verify.side_effect = \
            Exception("Fact check API error")
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        with pytest.raises(Exception, match="Fact check API error"):
            rag.query("What was revenue?")


class TestOrchestratorCorrectionContext:
    """Tests for correction context handling in the orchestrator."""
    
    def test_add_correction_context_empty_claims(self, mock_orchestrator_agents):
        """
        Test that empty unsupported_claims returns documents unchanged.
        
        Validates no unnecessary context is added when there are no issues.
        """
        from src.orchestrator import SelfCorrectingRAG
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        original_docs = [("Content 1", 0.9, "Reasoning 1")]
        result = rag._add_correction_context(original_docs, [])
        
        assert result == original_docs
    
    def test_add_correction_context_with_claims(self, mock_orchestrator_agents):
        """
        Test that correction context is properly prepended to documents.
        
        Validates that unsupported claims are formatted and added as
        high-priority context for the generator.
        """
        from src.orchestrator import SelfCorrectingRAG
        
        rag = SelfCorrectingRAG(
            vector_store=mock_orchestrator_agents["vector_store"],
            max_corrections=2
        )
        
        original_docs = [("Content 1", 0.9, "Reasoning 1")]
        claims = ["Claim A is false", "Claim B not in sources"]
        
        result = rag._add_correction_context(original_docs, claims)
        
        # Should have one more document (correction context)
        assert len(result) == len(original_docs) + 1
        
        # Correction context should be first
        correction_doc = result[0]
        assert "CORRECTION CONTEXT" in correction_doc[0]
        assert "Claim A is false" in correction_doc[0]
        assert "Claim B not in sources" in correction_doc[0]
        assert correction_doc[1] == 1.0  # High relevance
        
        # Original docs should follow
        assert result[1:] == original_docs


class TestInsufficientDocumentsError:
    """Tests for InsufficientDocumentsError exception."""
    
    def test_exception_stores_counts(self):
        """
        Test that exception stores retrieval and filter counts.
        
        Validates that debugging information is preserved.
        """
        from src.orchestrator import InsufficientDocumentsError
        
        exc = InsufficientDocumentsError(
            message="Not enough docs",
            docs_retrieved=10,
            docs_after_filter=0
        )
        
        assert exc.docs_retrieved == 10
        assert exc.docs_after_filter == 0
        assert "Not enough docs" in str(exc)

