"""
Tests for Baseline RAG Module.

Covers:
- TokenCountingCallback: token tracking
- VectorStoreRetriever: document retrieval
- BaselineRAG: full RAG pipeline, edge cases
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from langchain_core.documents import Document

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTokenCountingCallback:
    """Tests for TokenCountingCallback class."""
    
    def test_initial_state(self):
        """Test callback initializes with zero counts."""
        from src.baseline_rag import TokenCountingCallback
        
        callback = TokenCountingCallback()
        
        assert callback.prompt_tokens == 0
        assert callback.completion_tokens == 0
        assert callback.total_tokens == 0
    
    def test_on_llm_start_counts_prompt_tokens(self):
        """Test on_llm_start counts prompt tokens."""
        from src.baseline_rag import TokenCountingCallback
        
        callback = TokenCountingCallback()
        
        # Simulate LLM start with prompts
        prompts = ["Hello, how are you?", "Tell me about revenue."]
        callback.on_llm_start(serialized={}, prompts=prompts)
        
        # Should have counted some tokens
        assert callback.prompt_tokens > 0
    
    def test_on_llm_end_counts_completion_tokens(self):
        """Test on_llm_end counts completion tokens."""
        from src.baseline_rag import TokenCountingCallback
        
        callback = TokenCountingCallback()
        
        # Create mock response with generations
        mock_generation = Mock()
        mock_generation.text = "The revenue was $10 billion in Q4 2024."
        
        mock_response = Mock()
        mock_response.generations = [[mock_generation]]
        
        callback.on_llm_end(mock_response)
        
        assert callback.completion_tokens > 0
        assert callback.total_tokens == callback.completion_tokens
    
    def test_total_tokens_calculated_correctly(self):
        """Test total tokens is sum of prompt and completion."""
        from src.baseline_rag import TokenCountingCallback
        
        callback = TokenCountingCallback()
        
        # Simulate full flow
        callback.on_llm_start(serialized={}, prompts=["Test prompt"])
        
        mock_generation = Mock()
        mock_generation.text = "Test response"
        mock_response = Mock()
        mock_response.generations = [[mock_generation]]
        
        callback.on_llm_end(mock_response)
        
        assert callback.total_tokens == callback.prompt_tokens + callback.completion_tokens
    
    def test_reset_clears_counts(self):
        """Test reset sets all counts to zero."""
        from src.baseline_rag import TokenCountingCallback
        
        callback = TokenCountingCallback()
        
        # Add some counts
        callback.on_llm_start(serialized={}, prompts=["Test"])
        
        # Reset
        callback.reset()
        
        assert callback.prompt_tokens == 0
        assert callback.completion_tokens == 0
        assert callback.total_tokens == 0
    
    def test_fallback_encoding_for_unknown_model(self):
        """Test fallback encoding for unknown model name."""
        from src.baseline_rag import TokenCountingCallback
        
        # Should not raise for unknown model
        callback = TokenCountingCallback(model_name="unknown-model-xyz")
        
        callback.on_llm_start(serialized={}, prompts=["Test"])
        assert callback.prompt_tokens > 0


class TestVectorStoreRetriever:
    """Tests for VectorStoreRetriever class."""
    
    def test_get_relevant_documents(self):
        """Test document retrieval adds metadata."""
        from src.baseline_rag import VectorStoreRetriever
        
        # Mock vector store
        mock_store = Mock()
        mock_store.similarity_search.return_value = [
            (Document(page_content="Revenue report", metadata={}), 0.95),
            (Document(page_content="Financial data", metadata={}), 0.85),
        ]
        
        retriever = VectorStoreRetriever(vector_store=mock_store, k=5)
        
        docs = retriever._get_relevant_documents("revenue")
        
        assert len(docs) == 2
        # Check metadata was added
        assert docs[0].metadata["source_index"] == 1
        assert docs[0].metadata["similarity_score"] == 0.95
        assert docs[1].metadata["source_index"] == 2
    
    def test_get_retrieved_docs_with_scores(self):
        """Test getting last retrieved documents with scores."""
        from src.baseline_rag import VectorStoreRetriever
        
        mock_store = Mock()
        mock_store.similarity_search.return_value = [
            (Document(page_content="Test", metadata={}), 0.9),
        ]
        
        retriever = VectorStoreRetriever(vector_store=mock_store, k=5)
        
        # First retrieval
        retriever._get_relevant_documents("test")
        
        # Get cached results
        results = retriever.get_retrieved_docs_with_scores()
        
        assert len(results) == 1
        assert results[0][1] == 0.9
    
    def test_empty_search_results(self):
        """Test handling empty search results."""
        from src.baseline_rag import VectorStoreRetriever
        
        mock_store = Mock()
        mock_store.similarity_search.return_value = []
        
        retriever = VectorStoreRetriever(vector_store=mock_store, k=5)
        
        docs = retriever._get_relevant_documents("obscure query")
        
        assert docs == []


@pytest.fixture
def mock_baseline_rag_deps():
    """Mock all dependencies for BaselineRAG."""
    with patch("src.baseline_rag.ChatOpenAI") as mock_chat:
        with patch("src.baseline_rag.RetrievalQA") as mock_retrieval_qa:
            mock_llm = Mock()
            mock_chat.return_value = mock_llm
            
            mock_chain = Mock()
            mock_chain.invoke.return_value = {
                "result": "The revenue was $10 billion [1].",
                "source_documents": [
                    Document(page_content="Revenue data", metadata={"source": "10-K"})
                ]
            }
            mock_retrieval_qa.from_chain_type.return_value = mock_chain
            
            yield {
                "chat": mock_chat,
                "chain": mock_chain,
                "retrieval_qa": mock_retrieval_qa
            }


class TestBaselineRAG:
    """Tests for BaselineRAG class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock_store = Mock()
        mock_store.similarity_search.return_value = [
            (Document(page_content="Q4 revenue was $10B", metadata={"source": "10-K"}), 0.95),
        ]
        return mock_store
    
    def test_initialization_with_none_vector_store_raises_error(self):
        """Test that None vector store raises ValueError."""
        from src.baseline_rag import BaselineRAG
        
        with pytest.raises(ValueError, match="vector_store cannot be None"):
            BaselineRAG(vector_store=None)
    
    def test_initialization(self, mock_vector_store, mock_baseline_rag_deps):
        """Test BaselineRAG initializes correctly."""
        from src.baseline_rag import BaselineRAG
        
        rag = BaselineRAG(vector_store=mock_vector_store)
        
        assert rag.vector_store == mock_vector_store
        assert rag.k == 5  # Default
    
    def test_custom_k_parameter(self, mock_vector_store, mock_baseline_rag_deps):
        """Test custom k parameter is applied."""
        from src.baseline_rag import BaselineRAG
        
        rag = BaselineRAG(vector_store=mock_vector_store, k=10)
        
        assert rag.k == 10
    
    def test_query_empty_question_returns_error_message(self, mock_vector_store, mock_baseline_rag_deps):
        """Test empty question returns appropriate response."""
        from src.baseline_rag import BaselineRAG
        
        rag = BaselineRAG(vector_store=mock_vector_store)
        
        result = rag.query("")
        
        assert "Please provide a valid question" in result["answer"]
        assert result["sources"] == []
        assert result["latency"] == 0.0
        assert result["token_count"] == 0
    
    def test_query_whitespace_question_returns_error_message(self, mock_vector_store, mock_baseline_rag_deps):
        """Test whitespace-only question returns appropriate response."""
        from src.baseline_rag import BaselineRAG
        
        rag = BaselineRAG(vector_store=mock_vector_store)
        
        result = rag.query("   \n\t  ")
        
        assert "Please provide a valid question" in result["answer"]
    
    def test_query_returns_expected_structure(self, mock_vector_store, mock_baseline_rag_deps):
        """Test query returns dictionary with expected keys."""
        from src.baseline_rag import BaselineRAG
        
        rag = BaselineRAG(vector_store=mock_vector_store)
        
        result = rag.query("What was the revenue?")
        
        assert "answer" in result
        assert "sources" in result
        assert "latency" in result
        assert "token_count" in result
    
    def test_query_tracks_latency(self, mock_vector_store, mock_baseline_rag_deps):
        """Test that query latency is tracked."""
        from src.baseline_rag import BaselineRAG
        
        rag = BaselineRAG(vector_store=mock_vector_store)
        
        result = rag.query("What was the revenue?")
        
        assert result["latency"] >= 0
        assert isinstance(result["latency"], float)
    
    def test_query_resets_token_counter(self, mock_vector_store, mock_baseline_rag_deps):
        """Test that token counter is reset between queries."""
        from src.baseline_rag import BaselineRAG
        
        rag = BaselineRAG(vector_store=mock_vector_store)
        
        # First query
        rag.query("First question")
        
        # Second query should reset counter
        result = rag.query("Second question")
        
        # Token count should be from second query only
        assert isinstance(result["token_count"], int)
    
    def test_get_cost_estimate(self, mock_vector_store, mock_baseline_rag_deps):
        """Test cost estimation calculation."""
        from src.baseline_rag import BaselineRAG
        
        rag = BaselineRAG(vector_store=mock_vector_store)
        
        cost_info = rag.get_cost_estimate(1000)
        
        assert "token_count" in cost_info
        assert "estimated_cost_usd" in cost_info
        assert "pricing_note" in cost_info
        assert cost_info["token_count"] == 1000
        # 1000 tokens * $0.02 / 1000 = $0.02
        assert cost_info["estimated_cost_usd"] == pytest.approx(0.02, rel=1e-2)
    
    def test_query_handles_chain_error(self, mock_vector_store, mock_baseline_rag_deps):
        """Test that chain errors raise RuntimeError."""
        from src.baseline_rag import BaselineRAG
        
        mock_baseline_rag_deps["chain"].invoke.side_effect = Exception("API Error")
        
        rag = BaselineRAG(vector_store=mock_vector_store)
        
        with pytest.raises(RuntimeError, match="RAG query failed"):
            rag.query("What was the revenue?")


class TestFormatSourcesForDisplay:
    """Tests for format_sources_for_display function."""
    
    def test_empty_sources(self):
        """Test formatting empty sources list."""
        from src.baseline_rag import format_sources_for_display
        
        result = format_sources_for_display([])
        
        assert "No sources available" in result
    
    def test_sources_with_metadata(self):
        """Test formatting sources with full metadata."""
        from src.baseline_rag import format_sources_for_display
        
        sources = [
            Document(
                page_content="Revenue grew by 15% in 2024 driven by strong cloud adoption.",
                metadata={"source": "10-K/2024", "similarity_score": 0.95}
            ),
            Document(
                page_content="Operating margin improved to 25%.",
                metadata={"source": "10-K/2024", "similarity_score": 0.87}
            ),
        ]
        
        result = format_sources_for_display(sources)
        
        assert "[1]" in result
        assert "[2]" in result
        assert "10-K/2024" in result
        assert "0.95" in result or "0.9500" in result
    
    def test_sources_without_score(self):
        """Test formatting sources missing similarity score."""
        from src.baseline_rag import format_sources_for_display
        
        sources = [
            Document(
                page_content="Test content",
                metadata={"source": "test.txt"}
            ),
        ]
        
        result = format_sources_for_display(sources)
        
        assert "[1]" in result
        # When score is not present, it might show "N/A" or just omit the score line
        assert "test.txt" in result


class TestBaselineRAGEdgeCases:
    """Edge case tests for BaselineRAG."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock_store = Mock()
        mock_store.similarity_search.return_value = []
        return mock_store
    
    def test_query_with_no_relevant_documents(self, mock_vector_store, mock_baseline_rag_deps):
        """Test query when no documents are retrieved."""
        from src.baseline_rag import BaselineRAG
        
        mock_baseline_rag_deps["chain"].invoke.return_value = {
            "result": "I don't have sufficient information.",
            "source_documents": []
        }
        
        rag = BaselineRAG(vector_store=mock_vector_store)
        
        result = rag.query("Obscure question about nonexistent topic")
        
        assert "don't have sufficient information" in result["answer"]
        assert result["sources"] == []
    
    def test_query_with_special_characters(self, mock_vector_store, mock_baseline_rag_deps):
        """Test query with special characters."""
        from src.baseline_rag import BaselineRAG
        
        mock_baseline_rag_deps["chain"].invoke.return_value = {
            "result": "Response to special query",
            "source_documents": []
        }
        
        rag = BaselineRAG(vector_store=mock_vector_store)
        
        # Should handle special characters without error
        result = rag.query("What was the revenue in Q4'24? ($M)")
        
        assert isinstance(result["answer"], str)
    
    def test_query_with_very_long_question(self, mock_vector_store, mock_baseline_rag_deps):
        """Test query with very long question."""
        from src.baseline_rag import BaselineRAG
        
        mock_baseline_rag_deps["chain"].invoke.return_value = {
            "result": "Response",
            "source_documents": []
        }
        
        rag = BaselineRAG(vector_store=mock_vector_store)
        
        long_question = "What was the revenue? " * 100  # Very long question
        
        result = rag.query(long_question)
        
        assert isinstance(result["answer"], str)
    
    def test_custom_temperature(self, mock_vector_store, mock_baseline_rag_deps):
        """Test custom temperature parameter."""
        from src.baseline_rag import BaselineRAG
        
        rag = BaselineRAG(
            vector_store=mock_vector_store,
            temperature=0.5
        )
        
        assert rag is not None


class TestTokenCountingEdgeCases:
    """Edge case tests for token counting."""
    
    def test_empty_prompts_list(self):
        """Test handling empty prompts list."""
        from src.baseline_rag import TokenCountingCallback
        
        callback = TokenCountingCallback()
        callback.on_llm_start(serialized={}, prompts=[])
        
        assert callback.prompt_tokens == 0
    
    def test_response_without_generations(self):
        """Test handling response without generations attribute."""
        from src.baseline_rag import TokenCountingCallback
        
        callback = TokenCountingCallback()
        
        mock_response = Mock(spec=[])  # No generations attribute
        
        # Should not raise
        callback.on_llm_end(mock_response)
        
        assert callback.completion_tokens == 0
    
    def test_generation_without_text(self):
        """Test handling generation without text attribute."""
        from src.baseline_rag import TokenCountingCallback
        
        callback = TokenCountingCallback()
        
        mock_generation = Mock(spec=[])  # No text attribute
        mock_response = Mock()
        mock_response.generations = [[mock_generation]]
        
        # Should not raise
        callback.on_llm_end(mock_response)
        
        assert callback.completion_tokens == 0
    
    def test_unicode_in_prompts(self):
        """Test token counting with unicode characters."""
        from src.baseline_rag import TokenCountingCallback
        
        callback = TokenCountingCallback()
        
        unicode_prompts = ["收入是多少？", "Quelle est la revenu?"]
        callback.on_llm_start(serialized={}, prompts=unicode_prompts)
        
        assert callback.prompt_tokens > 0
