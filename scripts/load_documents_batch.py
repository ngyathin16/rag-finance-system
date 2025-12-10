#!/usr/bin/env python3
"""
Memory-efficient document loading for Render (512MB limit).
Processes documents in small batches to avoid OOM errors.
"""
import os
import sys
import pickle
from pathlib import Path
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import get_vector_store
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_documents_in_batches(
    pkl_path: str = "data/processed/documents.pkl",
    batch_size: int = 1000  # Small batches for memory efficiency
):
    """
    Load documents from pickle file into ChromaDB in small batches.
    Optimized for low-memory environments like Render Starter (512MB).
    
    Args:
        pkl_path: Path to the pickle file containing processed documents
        batch_size: Number of documents to process at once (default: 1000)
    """
    print("=" * 70)
    print("Memory-Efficient Document Loading for Render")
    print("=" * 70)
    
    # Check if pickle file exists
    if not os.path.exists(pkl_path):
        print(f"\n❌ Error: File not found: {pkl_path}")
        print("\nYou need to process documents first.")
        print("Options:")
        print("  1. Process locally, then upload data/ folder")
        print("  2. Use smaller dataset")
        return 1
    
    # Load documents count first (without loading all data)
    print(f"\n[1] Checking {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            documents = pickle.load(f)
        total_docs = len(documents)
        print(f"    ✅ Found {total_docs:,} documents")
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
    print(f"    Current documents: {stats['count']:,}")
    
    if stats['count'] > 0:
        user_input = input("\n⚠️  Collection already has documents. Continue adding? (y/n): ")
        if user_input.lower() != 'y':
            print("Aborted.")
            return 0
    
    # Process in batches
    print(f"\n[4] Adding {total_docs:,} documents in batches of {batch_size}...")
    print(f"    Total batches: {(total_docs + batch_size - 1) // batch_size}")
    print("    (This will take 15-20 minutes on Render Starter)")
    
    total_processed = 0
    failed_batches = 0
    
    for i in range(0, total_docs, batch_size):
        batch_num = (i // batch_size) + 1
        end_idx = min(i + batch_size, total_docs)
        batch = documents[i:end_idx]
        
        print(f"\n    Batch {batch_num}: Processing docs {i+1:,} to {end_idx:,}...")
        
        try:
            # Add batch to vector store
            vector_store.add_documents(batch, batch_size=100)
            total_processed += len(batch)
            print(f"    ✅ Batch {batch_num} complete ({total_processed:,}/{total_docs:,})")
            
            # Force garbage collection to free memory
            del batch
            gc.collect()
            
        except Exception as e:
            print(f"    ❌ Batch {batch_num} failed: {e}")
            failed_batches += 1
            
            if failed_batches >= 3:
                print("\n❌ Too many failures. Stopping.")
                break
    
    # Get final stats
    print("\n[5] Final collection stats:")
    final_stats = vector_store.get_collection_stats()
    print(f"    Collection: {final_stats['name']}")
    print(f"    Total documents: {final_stats['count']:,}")
    
    # Get cost estimate
    cost_stats = vector_store.cost_tracker.get_stats()
    print("\n[6] Embedding cost estimate:")
    print(f"    Total embeddings: {cost_stats['total_embeddings']:,}")
    print(f"    Estimated tokens: {cost_stats['total_tokens']:,}")
    print(f"    Estimated cost: ${cost_stats['total_cost_usd']:.4f}")
    
    if failed_batches > 0:
        print(f"\n⚠️  WARNING: {failed_batches} batches failed")
        print(f"    Successfully processed: {total_processed:,}/{total_docs:,}")
    else:
        print("\n" + "=" * 70)
        print("✅ SUCCESS! All documents loaded into vector store")
        print("=" * 70)
    
    return 0


if __name__ == "__main__":
    # Use smaller batch size for Render's limited memory
    exit_code = load_documents_in_batches(batch_size=500)
    sys.exit(exit_code)

