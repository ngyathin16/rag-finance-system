#!/usr/bin/env python3
"""
Load processed documents from .pkl file into ChromaDB vector store.
"""
import os
import sys
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import get_vector_store
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_documents_into_vector_store(pkl_path: str = "data/processed/documents.pkl"):
    """
    Load documents from pickle file into ChromaDB.
    
    Args:
        pkl_path: Path to the pickle file containing processed documents
    """
    print("=" * 70)
    print("Loading Documents into ChromaDB Vector Store")
    print("=" * 70)
    
    # Check if pickle file exists
    if not os.path.exists(pkl_path):
        print(f"\n❌ Error: File not found: {pkl_path}")
        print("\nRun this first:")
        print("  python scripts/process_documents.py")
        return 1
    
    # Load documents from pickle
    print(f"\n[1] Loading documents from {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            documents = pickle.load(f)
        print(f"    ✅ Loaded {len(documents)} documents")
    except Exception as e:
        print(f"    ❌ Failed to load pickle file: {e}")
        return 1
    
    # Initialize vector store
    print("\n[2] Initializing ChromaDB vector store...")
    try:
        vector_store = get_vector_store(mode="chroma")
        print("    ✅ Vector store initialized")
    except Exception as e:
        print(f"    ❌ Failed to initialize vector store: {e}")
        return 1
    
    # Check current collection stats
    print("\n[3] Checking current collection...")
    stats = vector_store.get_collection_stats()
    print(f"    Collection: {stats['name']}")
    print(f"    Current documents: {stats['count']}")
    
    # Add documents to vector store
    print(f"\n[4] Adding {len(documents)} documents to vector store...")
    print("    (This may take a few minutes...)")
    try:
        vector_store.add_documents(documents, batch_size=100)
        print("    ✅ Documents added successfully!")
    except Exception as e:
        print(f"    ❌ Failed to add documents: {e}")
        return 1
    
    # Get final stats
    print("\n[5] Final collection stats:")
    final_stats = vector_store.get_collection_stats()
    print(f"    Collection: {final_stats['name']}")
    print(f"    Total documents: {final_stats['count']}")
    
    # Get cost estimate
    cost_stats = vector_store.cost_tracker.get_stats()
    print("\n[6] Embedding cost estimate:")
    print(f"    Total embeddings: {cost_stats['total_embeddings']}")
    print(f"    Estimated tokens: {cost_stats['total_tokens']:,}")
    print(f"    Estimated cost: ${cost_stats['total_cost_usd']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS! Documents are now in the vector store")
    print("=" * 70)
    print("\nYou can now query the system:")
    print("  uvicorn src.api.main:app --reload")
    print("\nOr test directly:")
    print("  python src/baseline_rag.py")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit_code = load_documents_into_vector_store()
    sys.exit(exit_code)

