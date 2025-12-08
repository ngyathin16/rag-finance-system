# RAG Finance System

> Production-grade multi-agent RAG system for financial document Q&A with self-correction and comprehensive observability

## 🎯 Overview

A sophisticated Retrieval-Augmented Generation (RAG) system designed for querying financial documents (SEC filings, annual reports, etc.) with built-in self-correction, fact-checking, and production monitoring capabilities.

### Key Features

- **Multi-Agent Architecture:** Specialized agents for retrieval, relevance scoring, generation, and fact-checking
- **Self-Correction Loop:** Automatic validation and regeneration of incorrect answers
- **Production Observability:** OpenTelemetry integration with Jaeger tracing and Prometheus metrics
- **Cost Tracking:** Real-time token usage and cost monitoring for all LLM calls
- **Structured Outputs:** Type-safe Pydantic models for all agent responses
- **Async API:** FastAPI endpoints with async/await for optimal performance

## 🏗️ Architecture

```
Query → Query Processor → Retrieval Pipeline → Relevance Agent
                                                      ↓
                                            Generator Agent
                                                      ↓
                                            Fact-Check Agent
                                                      ↓
                                          Self-Correction Loop
                                                      ↓
                                              Final Answer
```

### Agent Pipeline

1. **Query Processor:** Intent classification and query rewriting
2. **Retrieval Pipeline:** Hybrid vector + keyword search
3. **Relevance Agent:** Filters and scores retrieved documents (top 3-5)
4. **Generator Agent:** Synthesizes answer with citations
5. **Fact-Check Agent:** Validates claims against source documents
6. **Self-Correction:** Triggers regeneration if validation fails (max 2 retries)

## 🚀 Setup

### Prerequisites

- Python 3.10+
- OpenAI API key
- (Optional) Pinecone account for production vector store
- (Optional) Docker for observability stack (Jaeger, Prometheus, Grafana)

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd rag-finance-system
```

2. **Create and activate virtual environment**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your API keys
# OPENAI_API_KEY=your_openai_key_here
# PINECONE_API_KEY=your_pinecone_key_here (optional)
```

5. **Run the API server**

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run with coverage
pytest --cov=src tests/
```

## 📊 Tech Stack

- **LLM Framework:** LangChain with structured outputs
- **Vector Store:** ChromaDB (local), Pinecone (production)
- **Embeddings:** OpenAI text-embedding-3-small
- **LLM Models:**
  - GPT-4o-mini (relevance scoring & fact-checking)
  - GPT-4-turbo-preview (answer generation)
- **Observability:** OpenTelemetry + Jaeger + Prometheus + Grafana
- **API:** FastAPI with async endpoints
- **Testing:** pytest, pytest-asyncio, pytest-mock

---

## 📦 Component Documentation

### Vector Store (`src/vector_store.py`)

The Vector Store module provides two implementations for storing and searching document embeddings.

#### EmbeddingCostTracker

Tracks embedding generation costs and counts for monitoring and budgeting.

```python
from src.vector_store import EmbeddingCostTracker

tracker = EmbeddingCostTracker()

# Log embedding generation
tracker.log_embeddings(count=100, estimated_tokens=50000)

# Get statistics
stats = tracker.get_stats()
print(f"Total embeddings: {stats['total_embeddings']}")
print(f"Estimated cost: ${stats['total_cost_usd']:.4f}")
```

#### ChromaVectorStore (Local Development)

Local vector store using ChromaDB with persistent storage.

```python
from src.vector_store import ChromaVectorStore
from langchain_core.documents import Document

# Initialize store
store = ChromaVectorStore(persist_directory="data/chroma_db")

# Add documents
documents = [
    Document(
        page_content="Q4 2024 revenue was $10 billion, up 15% YoY.",
        metadata={"id": "doc_1", "source": "10-K", "company": "TechCorp"}
    ),
    Document(
        page_content="Operating margin improved to 22% in fiscal 2024.",
        metadata={"id": "doc_2", "source": "10-K", "company": "TechCorp"}
    ),
]
store.add_documents(documents, batch_size=100)

# Similarity search
results = store.similarity_search("What was the revenue?", k=5)
for doc, score in results:
    print(f"Score: {score:.4f} - {doc.page_content[:100]}...")

# Hybrid search (vector + keyword boosting)
results = store.hybrid_search("Q4 revenue growth", k=5)

# Get collection statistics
stats = store.get_collection_stats()
print(f"Collection: {stats['name']}, Documents: {stats['count']}")
```

#### PineconeVectorStore (Production)

Production-grade vector store using Pinecone serverless.

```python
from src.vector_store import PineconeVectorStore

# Initialize (requires PINECONE_API_KEY env var)
store = PineconeVectorStore(
    index_name="financial-docs",
    cloud="aws",
    region="us-east-1"
)

# Add documents
store.add_documents(documents)

# Search with metadata filtering
results = store.similarity_search(
    "revenue growth",
    k=10,
    filter={"company": "TechCorp", "doc_type": "10-K"}
)

# Get index statistics
stats = store.get_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
```

#### Factory Function

```python
from src.vector_store import get_vector_store

# Local development
store = get_vector_store(mode="chroma")

# Production
store = get_vector_store(mode="pinecone", index_name="my-index")
```

---

### Relevance Agent (`src/agents/relevance_agent.py`)

LLM-powered document relevance scoring for financial Q&A.

#### RelevanceScore Model

```python
from src.agents.relevance_agent import RelevanceScore

# Structured output from LLM
score = RelevanceScore(
    score=0.85,
    reasoning="Document contains specific Q4 2024 revenue figures matching the query."
)
print(f"Score: {score.score}, Reasoning: {score.reasoning}")
```

#### RelevanceAgent

```python
from src.agents.relevance_agent import RelevanceAgent

# Initialize agent
agent = RelevanceAgent(model_name="gpt-4o-mini")

# Documents from vector search (content, original_similarity_score)
documents = [
    ("Q4 2024 revenue was $10B, up 15% from Q4 2023.", 0.92),
    ("The company was founded in 1985 in California.", 0.78),
    ("Operating expenses increased by 8% YoY.", 0.85),
    ("Weather patterns affected agricultural output.", 0.72),
]

# Score and filter documents
results = agent.score_documents(
    query="What was the company's revenue in Q4 2024?",
    documents=documents,
    threshold=0.7  # Filter out low-relevance documents
)

# Results: List[(content, llm_score, reasoning)]
for content, score, reasoning in results:
    print(f"Score: {score:.2f}")
    print(f"Reasoning: {reasoning}")
    print(f"Content: {content[:100]}...")
    print("-" * 50)
```

**Scoring Criteria:**
- **1.0 (Perfect Match):** Direct answer with specific data points
- **0.7-0.9 (Highly Relevant):** Relevant context, may require inference
- **0.4-0.6 (Tangentially Related):** Related topics but not direct answer
- **0.0-0.3 (Not Relevant):** Unrelated to the query

---

### Baseline RAG (`src/baseline_rag.py`)

Complete RAG pipeline for financial document Q&A.

#### TokenCountingCallback

```python
from src.baseline_rag import TokenCountingCallback

# Track token usage during LLM calls
callback = TokenCountingCallback(model_name="gpt-4-turbo-preview")

# After LLM call
print(f"Prompt tokens: {callback.prompt_tokens}")
print(f"Completion tokens: {callback.completion_tokens}")
print(f"Total tokens: {callback.total_tokens}")

# Reset for next query
callback.reset()
```

#### BaselineRAG

```python
from src.vector_store import get_vector_store
from src.baseline_rag import BaselineRAG, format_sources_for_display

# Initialize vector store
vector_store = get_vector_store(mode="chroma")

# Initialize RAG system
rag = BaselineRAG(
    vector_store=vector_store,
    k=5,           # Number of documents to retrieve
    temperature=0.2 # LLM temperature
)

# Query the system
result = rag.query("What was Apple's revenue in Q4 2024?")

# Access results
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} documents")
print(f"Latency: {result['latency']:.2f} seconds")
print(f"Tokens used: {result['token_count']}")

# Format sources for display
print(format_sources_for_display(result['sources']))

# Get cost estimate
cost_info = rag.get_cost_estimate(result['token_count'])
print(f"Estimated cost: ${cost_info['estimated_cost_usd']:.4f}")
```

---

### SEC Data Ingester (`scripts/ingest_sec_data.py`)

Downloads and processes SEC EDGAR financial statement data sets.

#### SECDataIngester

```python
from scripts.ingest_sec_data import SECDataIngester

# Initialize ingester
ingester = SECDataIngester(
    output_dir="data/raw",
    timeout=60  # Request timeout in seconds
)

# Download single quarter
zip_path = ingester.download_quarter(year=2024, quarter=3)
print(f"Downloaded: {zip_path}")

# Extract and parse
df = ingester.extract_and_parse(zip_path)
print(f"Parsed {len(df):,} records")
print(f"Columns: {df.columns.tolist()}")

# Convenience method: download + parse in one step
df = ingester.ingest_quarter(2024, 3)

# Download multiple quarters
results = ingester.ingest_range(
    start_year=2023, start_quarter=1,
    end_year=2024, end_quarter=2
)
for quarter_key, df in results.items():
    if df is not None:
        print(f"{quarter_key}: {len(df):,} records")
```

#### CLI Usage

```bash
# Download Q3 2024 data
python scripts/ingest_sec_data.py --year 2024 --quarter 3

# Download a range of quarters
python scripts/ingest_sec_data.py \
    --start-year 2023 --start-quarter 1 \
    --end-year 2024 --end-quarter 2

# Download only (skip parsing)
python scripts/ingest_sec_data.py --year 2024 --quarter 3 --download-only

# Custom output directory with verbose logging
python scripts/ingest_sec_data.py --year 2024 --quarter 3 \
    --output-dir ./my_data --verbose
```

---

### Document Processor (`scripts/process_documents.py`)

Processes SEC 10-K filings into LangChain Document objects for RAG.

#### DocumentProcessor

```python
from scripts.process_documents import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(
    chunk_size=1000,    # Maximum chunk size in characters
    chunk_overlap=200   # Overlap between chunks
)

# Process single filing
documents = processor.process_sec_filing("data/raw/apple_10k_2024.txt")

print(f"Created {len(documents)} document chunks")
for doc in documents[:3]:
    print(f"Section: {doc.metadata['section']}")
    print(f"Company: {doc.metadata['company']}")
    print(f"Period: {doc.metadata['period']}")
    print(f"Chunk size: {doc.metadata['chunk_size']} chars")
    print(f"Content preview: {doc.page_content[:100]}...")
    print("-" * 50)

# Process entire directory
all_documents = processor.process_directory(
    input_dir="data/raw",
    output_path="data/processed/documents.pkl",
    file_patterns=["*.txt", "*.htm"]
)
print(f"Processed {len(all_documents)} total chunks")
```

#### CLI Usage

```bash
# Process all filings in data/raw/
python scripts/process_documents.py

# Process single file
python scripts/process_documents.py --file data/raw/apple_10k.txt

# Custom chunk settings
python scripts/process_documents.py --chunk-size 500 --chunk-overlap 100

# Custom input/output
python scripts/process_documents.py \
    --input-dir ./filings \
    --output ./processed/docs.pkl \
    --verbose
```

#### Extracted Sections

The processor extracts these key 10-K sections:
- **Item 1: Business** - Company description and operations
- **Item 1A: Risk Factors** - Material risks to the business
- **Item 7: MD&A** - Management's discussion and analysis
- **Item 8: Financial Statements** - Audited financial data

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_vector_store.py
pytest tests/agents/test_relevance_agent.py
pytest tests/test_baseline_rag.py
pytest tests/test_ingest_sec_data.py
pytest tests/test_process_documents.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov=scripts --cov-report=html tests/

# Run specific test class
pytest tests/test_vector_store.py::TestChromaVectorStore

# Run tests matching pattern
pytest -k "empty" -v  # All tests with "empty" in name
```

### Test Categories

| Module | Test File | Coverage |
|--------|-----------|----------|
| Vector Store | `tests/test_vector_store.py` | Cost tracking, document ops, search, edge cases |
| Relevance Agent | `tests/agents/test_relevance_agent.py` | Scoring, filtering, thresholds, error handling |
| Baseline RAG | `tests/test_baseline_rag.py` | Token counting, retrieval, full pipeline |
| SEC Ingester | `tests/test_ingest_sec_data.py` | Download, extraction, timeout handling |
| Doc Processor | `tests/test_process_documents.py` | Section extraction, chunking, metadata |

---

## 📝 Usage

### API Endpoints

**POST /query**

Submit a question about financial documents:

```json
{
  "query": "What was Apple's revenue in Q3 2024?",
  "max_retries": 2,
  "relevance_threshold": 0.7
}
```

**GET /health**

Health check endpoint for monitoring

### Python SDK (Coming Soon)

```python
from rag_finance import RAGSystem

rag = RAGSystem()
result = rag.query("What was Apple's revenue in Q3 2024?")
print(result.answer)
print(result.citations)
```

---

## 📈 Monitoring

### Metrics Tracked

- Query latency (p50, p95, p99)
- Token usage and cost per query
- Self-correction rate
- Verification status distribution
- Cache hit rate

### Observability Stack

Start the observability stack with Docker Compose:

```bash
docker-compose up -d
```

Access dashboards:
- **Jaeger UI:** http://localhost:16686 (distributed tracing)
- **Prometheus:** http://localhost:9090 (metrics)
- **Grafana:** http://localhost:3000 (dashboards)

---

## 🗂️ Project Structure

```
rag-finance-system/
├── src/
│   ├── agents/           # Agent implementations
│   │   └── relevance_agent.py
│   ├── api/              # FastAPI endpoints
│   ├── observability/    # OpenTelemetry setup
│   ├── monitoring/       # Metrics and cost tracking
│   ├── baseline_rag.py   # RAG pipeline implementation
│   └── vector_store.py   # Vector store implementations
├── scripts/
│   ├── ingest_sec_data.py    # SEC data downloader
│   └── process_documents.py  # Document processor
├── data/
│   ├── raw/              # Raw financial documents
│   ├── processed/        # Processed/chunked documents
│   └── chroma_db/        # Local vector store
├── tests/
│   ├── agents/           # Agent unit tests
│   ├── integration/      # End-to-end tests
│   ├── evaluation/       # FinQA dataset evaluation
│   ├── test_vector_store.py
│   ├── test_baseline_rag.py
│   ├── test_ingest_sec_data.py
│   └── test_process_documents.py
├── config/               # Configuration files
└── requirements.txt      # Python dependencies
```

---

## 🎯 Roadmap

- [ ] Complete multi-agent pipeline implementation
- [ ] Add evaluation on FinQA dataset
- [ ] Implement caching layer for repeated queries
- [ ] Add streaming responses for long answers
- [ ] Deploy to Railway/Render
- [ ] Create demo video

---

## 📚 Documentation

- [Architecture Details](docs/architecture.md) (Coming Soon)
- [API Reference](docs/api.md) (Coming Soon)
- [Agent Design](docs/agents.md) (Coming Soon)

---

## 🤝 Contributing

This is a portfolio project, but suggestions and feedback are welcome!

---

## 📄 License

MIT License

---

**Built for demonstrating production-grade RAG systems for senior AI/ML engineering roles**

*Last updated: 2025-12-08*
