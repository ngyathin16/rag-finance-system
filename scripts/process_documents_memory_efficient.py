#!/usr/bin/env python3
"""
Memory-efficient document processor for Render (512MB limit).
Processes files in batches and streams directly to ChromaDB.
"""
import os
import sys
import logging
from pathlib import Path
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.process_documents import DocumentProcessor
from src.vector_store import get_vector_store
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_and_load_streaming(
    input_dir: str = "data/raw/2024q3",
    batch_size: int = 50,  # Process 50 files at a time
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    Process documents and load directly into ChromaDB in batches.
    Memory-efficient for Render Starter (512MB).
    
    Args:
        input_dir: Directory containing SEC filings
        batch_size: Number of files to process before loading to DB
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    print("=" * 70)
    print("Memory-Efficient Document Processing for Render")
    print("=" * 70)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"\n❌ Error: Directory not found: {input_dir}")
        print("\nMake sure you've downloaded data first:")
        print("  python scripts/ingest_sec_data.py --year 2024 --quarter 3")
        return 1
    
    # Find all files
    print(f"\n[1] Scanning {input_dir} for files...")
    file_patterns = ["*.txt"]
    files = []
    for pattern in file_patterns:
        files.extend(input_path.rglob(pattern))
    
    files = sorted(set(files))
    
    if not files:
        print(f"    ❌ No files found")
        return 1
    
    print(f"    ✅ Found {len(files)} files")
    
    # Initialize processor
    print(f"\n[2] Initializing document processor...")
    print(f"    Chunk size: {chunk_size}")
    print(f"    Chunk overlap: {chunk_overlap}")
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Initialize vector store
    print(f"\n[3] Initializing ChromaDB vector store...")
    try:
        vector_store = get_vector_store(mode="chroma")
        stats = vector_store.get_collection_stats()
        print(f"    ✅ Vector store initialized")
        print(f"    Current documents: {stats['count']:,}")
    except Exception as e:
        print(f"    ❌ Failed to initialize vector store: {e}")
        return 1
    
    # Process in batches
    print(f"\n[4] Processing and loading {len(files)} files in batches of {batch_size}...")
    print(f"    Total batches: {(len(files) + batch_size - 1) // batch_size}")
    print(f"    (This will take 20-30 minutes on Render)")
    
    total_documents = 0
    successful_files = 0
    failed_files = 0
    
    for batch_start in range(0, len(files), batch_size):
        batch_num = (batch_start // batch_size) + 1
        batch_end = min(batch_start + batch_size, len(files))
        batch_files = files[batch_start:batch_end]
        
        print(f"\n    Batch {batch_num}: Processing files {batch_start+1} to {batch_end}...")
        
        batch_documents = []
        
        # Process files in this batch
        for filepath in tqdm(batch_files, desc=f"    Processing batch {batch_num}", leave=False):
            try:
                documents = processor.process_sec_filing(str(filepath))
                batch_documents.extend(documents)
                successful_files += 1
            except Exception as e:
                logger.error(f"Failed to process {filepath.name}: {e}")
                failed_files += 1
        
        # Load batch into ChromaDB
        if batch_documents:
            try:
                print(f"    Loading {len(batch_documents)} documents into ChromaDB...")
                vector_store.add_documents(batch_documents, batch_size=100)
                total_documents += len(batch_documents)
                print(f"    ✅ Batch {batch_num} complete ({total_documents:,} total docs)")
                
                # Clear memory
                del batch_documents
                gc.collect()
                
            except Exception as e:
                print(f"    ❌ Failed to load batch {batch_num}: {e}")
                failed_files += len(batch_files)
    
    # Final stats
    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    
    final_stats = vector_store.get_collection_stats()
    print(f"\n[5] Final collection stats:")
    print(f"    Collection: {final_stats['name']}")
    print(f"    Total documents: {final_stats['count']:,}")
    print(f"    Files processed: {successful_files} successful, {failed_files} failed")
    
    # Cost estimate
    cost_stats = vector_store.cost_tracker.get_stats()
    print(f"\n[6] Embedding cost estimate:")
    print(f"    Total embeddings: {cost_stats['total_embeddings']:,}")
    print(f"    Estimated tokens: {cost_stats['total_tokens']:,}")
    print(f"    Estimated cost: ${cost_stats['total_cost_usd']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS! Documents are now searchable in production")
    print("=" * 70)
    print("\nTest your production API at:")
    print("  https://your-app.onrender.com")
    print("=" * 70)
    
    return 0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Memory-efficient document processor for Render"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/2024q3",
        help="Input directory (default: data/raw/2024q3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Files to process per batch (default: 50)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Text chunk size (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap (default: 200)"
    )
    
    args = parser.parse_args()
    
    exit_code = process_and_load_streaming(
        input_dir=args.input_dir,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

