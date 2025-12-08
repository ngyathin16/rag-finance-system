"""
Tests for Relevance Agent Module.

Covers:
- RelevanceScore: Pydantic model validation
- RelevanceAgent: document scoring, filtering, edge cases
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from pydantic import ValidationError

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRelevanceScore:
    """Tests for RelevanceScore Pydantic model."""
    
    def test_valid_score(self):
        """Test creating a valid RelevanceScore."""
        from src.agents.relevance_agent import RelevanceScore
        
        score = RelevanceScore(
            score=0.85,
            reasoning="Document contains specific Q4 2023 revenue figures"
        )
        
        assert score.score == 0.85
        assert "Q4 2023" in score.reasoning
    
    def test_score_at_boundaries(self):
        """Test scores at boundary values 0.0 and 1.0."""
        from src.agents.relevance_agent import RelevanceScore
        
        # Minimum score
        min_score = RelevanceScore(score=0.0, reasoning="Not relevant")
        assert min_score.score == 0.0
        
        # Maximum score
        max_score = RelevanceScore(score=1.0, reasoning="Perfect match")
        assert max_score.score == 1.0
    
    def test_score_below_zero_raises_error(self):
        """Test that score below 0 raises validation error."""
        from src.agents.relevance_agent import RelevanceScore
        
        with pytest.raises(ValidationError):
            RelevanceScore(score=-0.1, reasoning="Invalid")
    
    def test_score_above_one_raises_error(self):
        """Test that score above 1 raises validation error."""
        from src.agents.relevance_agent import RelevanceScore
        
        with pytest.raises(ValidationError):
            RelevanceScore(score=1.5, reasoning="Invalid")
    
    def test_reasoning_max_length(self):
        """Test that reasoning respects max_length constraint."""
        from src.agents.relevance_agent import RelevanceScore
        
        # Exactly at max length should work
        long_reasoning = "x" * 200
        score = RelevanceScore(score=0.5, reasoning=long_reasoning)
        assert len(score.reasoning) == 200
        
        # Above max length should fail
        with pytest.raises(ValidationError):
            RelevanceScore(score=0.5, reasoning="x" * 201)


@pytest.fixture
def mock_relevance_agent_deps():
    """Mock all dependencies for RelevanceAgent."""
    with patch("src.agents.relevance_agent.ChatOpenAI") as mock_chat:
        with patch("src.agents.relevance_agent.ChatPromptTemplate") as mock_prompt:
            mock_llm = Mock()
            mock_chain = Mock()
            
            # Create a mock prompt that supports | operator
            mock_prompt_instance = MagicMock()
            mock_prompt_instance.__or__ = Mock(return_value=mock_chain)
            mock_prompt.from_messages.return_value = mock_prompt_instance
            
            # Make the chain from | operator
            mock_llm.with_structured_output.return_value = mock_chain
            mock_chat.return_value = mock_llm
            
            yield {
                "chat": mock_chat,
                "prompt": mock_prompt,
                "llm": mock_llm,
                "chain": mock_chain
            }


class TestRelevanceAgent:
    """Tests for RelevanceAgent class."""
    
    def test_initialization(self, mock_relevance_agent_deps):
        """Test RelevanceAgent initializes correctly."""
        from src.agents.relevance_agent import RelevanceAgent
        
        agent = RelevanceAgent(model_name="gpt-4o-mini")
        
        assert agent.model_name == "gpt-4o-mini"
    
    def test_empty_model_name_raises_error(self, mock_relevance_agent_deps):
        """Test that empty model name raises ValueError."""
        from src.agents.relevance_agent import RelevanceAgent
        
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            RelevanceAgent(model_name="")
        
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            RelevanceAgent(model_name=None)
    
    def test_empty_query_raises_error(self, mock_relevance_agent_deps):
        """Test that empty query raises ValueError."""
        from src.agents.relevance_agent import RelevanceAgent
        
        agent = RelevanceAgent()
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            agent.score_documents(query="", documents=[])
        
        with pytest.raises(ValueError, match="query cannot be empty"):
            agent.score_documents(query="   ", documents=[])
    
    def test_invalid_threshold_raises_error(self, mock_relevance_agent_deps):
        """Test that invalid threshold raises ValueError."""
        from src.agents.relevance_agent import RelevanceAgent
        
        agent = RelevanceAgent()
        
        with pytest.raises(ValueError, match="threshold must be between"):
            agent.score_documents(
                query="test query",
                documents=[],
                threshold=-0.1
            )
        
        with pytest.raises(ValueError, match="threshold must be between"):
            agent.score_documents(
                query="test query",
                documents=[],
                threshold=1.5
            )
    
    def test_score_empty_documents_list(self, mock_relevance_agent_deps):
        """Test scoring empty document list returns empty list."""
        from src.agents.relevance_agent import RelevanceAgent
        
        agent = RelevanceAgent()
        
        results = agent.score_documents(
            query="What was the revenue?",
            documents=[],
            threshold=0.7
        )
        
        assert results == []
    
    def test_score_documents_filters_by_threshold(self, mock_relevance_agent_deps):
        """Test that documents below threshold are filtered out."""
        from src.agents.relevance_agent import RelevanceAgent, RelevanceScore
        
        agent = RelevanceAgent()
        
        # Mock chain to return different scores
        scores = [
            RelevanceScore(score=0.9, reasoning="Highly relevant"),
            RelevanceScore(score=0.5, reasoning="Tangentially related"),
            RelevanceScore(score=0.8, reasoning="Good match"),
        ]
        mock_relevance_agent_deps["chain"].invoke = Mock(side_effect=scores)
        agent.chain = mock_relevance_agent_deps["chain"]
        
        documents = [
            ("Revenue was $10B", 0.9),
            ("Company history", 0.7),
            ("Q4 financial data", 0.85),
        ]
        
        results = agent.score_documents(
            query="What was the revenue?",
            documents=documents,
            threshold=0.7
        )
        
        # Only scores >= 0.7 should pass (0.9 and 0.8)
        assert len(results) == 2
        assert all(score >= 0.7 for _, score, _ in results)
    
    def test_score_documents_sorted_by_score(self, mock_relevance_agent_deps):
        """Test that results are sorted by score descending."""
        from src.agents.relevance_agent import RelevanceAgent, RelevanceScore
        
        agent = RelevanceAgent()
        
        scores = [
            RelevanceScore(score=0.7, reasoning="Good"),
            RelevanceScore(score=0.9, reasoning="Excellent"),
            RelevanceScore(score=0.8, reasoning="Very good"),
        ]
        mock_relevance_agent_deps["chain"].invoke = Mock(side_effect=scores)
        agent.chain = mock_relevance_agent_deps["chain"]
        
        documents = [
            ("Doc 1", 0.7),
            ("Doc 2", 0.9),
            ("Doc 3", 0.8),
        ]
        
        results = agent.score_documents(
            query="test",
            documents=documents,
            threshold=0.5
        )
        
        # Results should be sorted by score descending
        scores_only = [score for _, score, _ in results]
        assert scores_only == sorted(scores_only, reverse=True)
    
    def test_score_documents_returns_max_5(self, mock_relevance_agent_deps):
        """Test that at most 5 documents are returned."""
        from src.agents.relevance_agent import RelevanceAgent, RelevanceScore
        
        agent = RelevanceAgent()
        
        # Create 10 documents all scoring above threshold
        scores = [RelevanceScore(score=0.9, reasoning="Relevant") for _ in range(10)]
        mock_relevance_agent_deps["chain"].invoke = Mock(side_effect=scores)
        agent.chain = mock_relevance_agent_deps["chain"]
        
        documents = [(f"Doc {i}", 0.9) for i in range(10)]
        
        results = agent.score_documents(
            query="test",
            documents=documents,
            threshold=0.5
        )
        
        assert len(results) <= 5
    
    def test_score_documents_handles_llm_error(self, mock_relevance_agent_deps):
        """Test that LLM errors fall back to original score."""
        from src.agents.relevance_agent import RelevanceAgent, RelevanceScore
        
        agent = RelevanceAgent()
        
        # First call succeeds, second fails
        def mock_invoke(params):
            if "error doc" in params.get("document", ""):
                raise Exception("LLM API error")
            return RelevanceScore(score=0.8, reasoning="Valid")
        
        mock_relevance_agent_deps["chain"].invoke = Mock(side_effect=mock_invoke)
        agent.chain = mock_relevance_agent_deps["chain"]
        
        documents = [
            ("Good document", 0.85),
            ("error doc", 0.75),
        ]
        
        results = agent.score_documents(
            query="test",
            documents=documents,
            threshold=0.5
        )
        
        # Both documents should be returned
        assert len(results) == 2
        
        # Find the error doc result
        error_result = next((r for r in results if "error doc" in r[0]), None)
        assert error_result is not None
        # Should use original score as fallback
        assert error_result[1] == 0.75
        assert "fallback" in error_result[2].lower() or "failed" in error_result[2].lower()
    
    def test_low_score_documents_filtered(self, mock_relevance_agent_deps):
        """Test that very low scoring documents are filtered out."""
        from src.agents.relevance_agent import RelevanceAgent, RelevanceScore
        
        agent = RelevanceAgent()
        
        # All documents score very low
        scores = [
            RelevanceScore(score=0.1, reasoning="Not relevant"),
            RelevanceScore(score=0.2, reasoning="Unrelated"),
            RelevanceScore(score=0.15, reasoning="Off topic"),
        ]
        mock_relevance_agent_deps["chain"].invoke = Mock(side_effect=scores)
        agent.chain = mock_relevance_agent_deps["chain"]
        
        documents = [
            ("Weather report", 0.5),
            ("Sports news", 0.4),
            ("Recipe blog", 0.3),
        ]
        
        results = agent.score_documents(
            query="What was Apple's revenue?",
            documents=documents,
            threshold=0.7  # High threshold
        )
        
        # All should be filtered out
        assert len(results) == 0
    
    def test_threshold_boundary_values(self, mock_relevance_agent_deps):
        """Test threshold at exact boundary values."""
        from src.agents.relevance_agent import RelevanceAgent, RelevanceScore
        
        agent = RelevanceAgent()
        
        scores = [
            RelevanceScore(score=0.7, reasoning="Exactly at threshold"),
            RelevanceScore(score=0.69, reasoning="Just below threshold"),
        ]
        mock_relevance_agent_deps["chain"].invoke = Mock(side_effect=scores)
        agent.chain = mock_relevance_agent_deps["chain"]
        
        documents = [
            ("Doc 1", 0.7),
            ("Doc 2", 0.69),
        ]
        
        results = agent.score_documents(
            query="test",
            documents=documents,
            threshold=0.7
        )
        
        # Only the document exactly at threshold should pass
        assert len(results) == 1
        assert results[0][1] == 0.7


class TestRelevanceAgentEdgeCases:
    """Edge case tests for RelevanceAgent."""
    
    def test_very_long_document(self, mock_relevance_agent_deps):
        """Test handling very long document content."""
        from src.agents.relevance_agent import RelevanceAgent, RelevanceScore
        
        agent = RelevanceAgent()
        
        mock_relevance_agent_deps["chain"].invoke = Mock(
            return_value=RelevanceScore(score=0.8, reasoning="Relevant")
        )
        agent.chain = mock_relevance_agent_deps["chain"]
        
        # Create a very long document
        long_content = "Financial data. " * 10000  # ~160KB of text
        documents = [(long_content, 0.9)]
        
        results = agent.score_documents(
            query="financial data",
            documents=documents,
            threshold=0.5
        )
        
        assert len(results) == 1
    
    def test_document_with_special_characters(self, mock_relevance_agent_deps):
        """Test handling documents with special characters."""
        from src.agents.relevance_agent import RelevanceAgent, RelevanceScore
        
        agent = RelevanceAgent()
        
        mock_relevance_agent_deps["chain"].invoke = Mock(
            return_value=RelevanceScore(score=0.8, reasoning="Valid")
        )
        agent.chain = mock_relevance_agent_deps["chain"]
        
        # Document with special characters
        special_doc = "Revenue: $4.2B (FY2024) - 15% YoY growth\n\t• Item 1\n\t• Item 2"
        documents = [(special_doc, 0.9)]
        
        results = agent.score_documents(
            query="revenue growth",
            documents=documents,
            threshold=0.5
        )
        
        assert len(results) == 1
    
    def test_unicode_content(self, mock_relevance_agent_deps):
        """Test handling unicode characters in documents."""
        from src.agents.relevance_agent import RelevanceAgent, RelevanceScore
        
        agent = RelevanceAgent()
        
        mock_relevance_agent_deps["chain"].invoke = Mock(
            return_value=RelevanceScore(score=0.8, reasoning="Valid")
        )
        agent.chain = mock_relevance_agent_deps["chain"]
        
        # Unicode content
        unicode_doc = "收入增长了15% • Résumé • über • café"
        documents = [(unicode_doc, 0.9)]
        
        results = agent.score_documents(
            query="test",
            documents=documents,
            threshold=0.5
        )
        
        assert len(results) == 1
    
    def test_empty_document_content(self, mock_relevance_agent_deps):
        """Test handling document with empty content."""
        from src.agents.relevance_agent import RelevanceAgent, RelevanceScore
        
        agent = RelevanceAgent()
        
        mock_relevance_agent_deps["chain"].invoke = Mock(
            return_value=RelevanceScore(score=0.1, reasoning="Empty content")
        )
        agent.chain = mock_relevance_agent_deps["chain"]
        
        # Empty document content
        documents = [("", 0.9)]
        
        results = agent.score_documents(
            query="test",
            documents=documents,
            threshold=0.5
        )
        
        # Should process without error, score should be low
        assert len(results) == 0  # Filtered by threshold


class TestOpenTelemetryIntegration:
    """Tests for OpenTelemetry instrumentation."""
    
    def test_span_created_for_scoring(self, mock_relevance_agent_deps):
        """Test that OpenTelemetry span is created for score_documents."""
        from src.agents.relevance_agent import RelevanceAgent, RelevanceScore
        
        with patch("src.agents.relevance_agent.tracer") as mock_tracer:
            mock_span = Mock()
            mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=False)
            
            agent = RelevanceAgent()
            mock_relevance_agent_deps["chain"].invoke = Mock(
                return_value=RelevanceScore(score=0.8, reasoning="Valid")
            )
            agent.chain = mock_relevance_agent_deps["chain"]
            
            documents = [("Test doc", 0.9)]
            agent.score_documents(query="test", documents=documents, threshold=0.5)
            
            # Verify span was created
            mock_tracer.start_as_current_span.assert_called_once_with(
                "relevance_agent.score_documents"
            )
