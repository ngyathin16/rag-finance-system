#!/usr/bin/env python3
"""
SEC 10-K Document Processor for RAG Finance System.

This script processes SEC 10-K filings by extracting key sections,
chunking text, and creating LangChain Document objects with metadata
for use in retrieval-augmented generation pipelines.
"""

import argparse
import logging
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class DocumentProcessorError(Exception):
    """Base exception for document processing errors."""

    pass


class MalformedFilingError(DocumentProcessorError):
    """Exception raised when a filing is malformed or cannot be parsed."""

    pass


class SectionExtractionError(DocumentProcessorError):
    """Exception raised when section extraction fails."""

    pass


class DocumentProcessor:
    """
    Processes SEC 10-K filings into LangChain Document objects.

    This class provides methods to extract key sections from SEC 10-K filings,
    chunk the text using LangChain's RecursiveCharacterTextSplitter, and create
    Document objects with rich metadata for downstream RAG applications.

    Attributes:
        chunk_size: Maximum size of text chunks.
        chunk_overlap: Overlap between consecutive chunks.
        text_splitter: LangChain text splitter instance.

    Example:
        >>> processor = DocumentProcessor()
        >>> documents = processor.process_sec_filing("data/raw/10k_filing.txt")
        >>> print(len(documents))
        42
    """

    # SEC 10-K section patterns for extraction
    SECTION_PATTERNS = {
        "Item 1: Business": [
            r"(?i)item\s*1\.?\s*[-–—]?\s*business",
            r"(?i)item\s*1\b[^0-9a-zA-Z].*?business",
        ],
        "Item 1A: Risk Factors": [
            r"(?i)item\s*1a\.?\s*[-–—]?\s*risk\s*factors?",
            r"(?i)item\s*1a\b[^0-9a-zA-Z].*?risk",
        ],
        "Item 7: MD&A": [
            r"(?i)item\s*7\.?\s*[-–—]?\s*management['']?s?\s*discussion",
            r"(?i)item\s*7\b[^0-9a-zA-Z].*?discussion\s*and\s*analysis",
            r"(?i)item\s*7\.?\s*[-–—]?\s*md\s*&\s*a",
        ],
        "Item 8: Financial Statements": [
            r"(?i)item\s*8\.?\s*[-–—]?\s*financial\s*statements?",
            r"(?i)item\s*8\b[^0-9a-zA-Z].*?financial",
        ],
    }

    # Patterns to identify section boundaries (next items)
    NEXT_SECTION_PATTERNS = [
        r"(?i)item\s*1a\b",
        r"(?i)item\s*1b\b",
        r"(?i)item\s*2\b",
        r"(?i)item\s*3\b",
        r"(?i)item\s*4\b",
        r"(?i)item\s*5\b",
        r"(?i)item\s*6\b",
        r"(?i)item\s*7\b",
        r"(?i)item\s*7a\b",
        r"(?i)item\s*8\b",
        r"(?i)item\s*9\b",
        r"(?i)item\s*9a\b",
        r"(?i)item\s*9b\b",
        r"(?i)item\s*10\b",
        r"(?i)item\s*11\b",
        r"(?i)item\s*12\b",
        r"(?i)item\s*13\b",
        r"(?i)item\s*14\b",
        r"(?i)item\s*15\b",
        r"(?i)part\s*(ii|iii|iv)\b",
        r"(?i)signatures?\b",
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the Document Processor.

        Args:
            chunk_size: Maximum size of text chunks (default: 1000).
            chunk_overlap: Number of overlapping characters between chunks (default: 200).
            separators: List of separators for text splitting.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", " "]

        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"Initialized DocumentProcessor with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def _extract_company_name(self, text: str, filepath: str) -> str:
        """
        Extract company name from filing text or filename.

        Args:
            text: The filing text content.
            filepath: Path to the filing file.

        Returns:
            Extracted company name or 'Unknown'.
        """
        # Try to extract from common SEC filing header patterns
        patterns = [
            r"(?i)company\s*name[:\s]+([^\n]+)",
            r"(?i)registrant[:\s]+([^\n]+)",
            r"(?i)(?:form\s+10-k[^\n]*\n+)([A-Z][A-Z\s&,.'()-]+(?:INC|CORP|LLC|LTD|CO|COMPANY|CORPORATION)?)",
            r"(?i)annual\s+report[^\n]*\n+([A-Z][A-Z\s&,.'()-]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:5000])  # Search only in header
            if match:
                company = match.group(1).strip()
                # Clean up the company name
                company = re.sub(r"\s+", " ", company)
                company = company.strip(".,;:")
                if len(company) > 3 and len(company) < 100:
                    return company

        # Fallback: try to extract from filename
        filename = Path(filepath).stem
        # Remove common patterns from filename
        clean_name = re.sub(r"[-_]?(10k|10-k|annual|report|filing)[-_]?", "", filename, flags=re.IGNORECASE)
        clean_name = re.sub(r"[-_]?\d{4}[-_]?", "", clean_name)
        clean_name = clean_name.replace("_", " ").replace("-", " ").strip()

        if clean_name:
            return clean_name.title()

        return "Unknown"

    def _extract_period(self, text: str, filepath: str) -> str:
        """
        Extract fiscal period from filing text or filename.

        Args:
            text: The filing text content.
            filepath: Path to the filing file.

        Returns:
            Extracted period (e.g., 'FY2024', 'Q4 2024') or 'Unknown'.
        """
        # Try to extract fiscal year from text
        patterns = [
            r"(?i)fiscal\s*year\s*(?:ended?|ending)?\s*(?:\w+\s*\d{1,2},?\s*)?(\d{4})",
            r"(?i)for\s*the\s*(?:fiscal\s*)?year\s*ended?\s*(?:\w+\s*\d{1,2},?\s*)?(\d{4})",
            r"(?i)annual\s*report\s*(?:for\s*)?(?:the\s*)?(?:fiscal\s*)?year\s*(\d{4})",
            r"(?i)form\s*10-k\s*(?:for\s*)?(?:the\s*)?(?:fiscal\s*)?year\s*(?:ended?\s*)?(?:\w+\s*\d{1,2},?\s*)?(\d{4})",
            r"(?i)(?:december|january|february|march|april|may|june|july|august|september|october|november)\s*\d{1,2},?\s*(\d{4})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:10000])  # Search in header area
            if match:
                year = match.group(1)
                return f"FY{year}"

        # Try to extract from filename
        filename = Path(filepath).stem
        year_match = re.search(r"(\d{4})", filename)
        if year_match:
            return f"FY{year_match.group(1)}"

        # Try quarter pattern
        quarter_match = re.search(r"[qQ](\d)\s*(\d{4})", text[:10000])
        if quarter_match:
            return f"Q{quarter_match.group(1)} {quarter_match.group(2)}"

        return "Unknown"

    def _find_section_start(self, text: str, patterns: list[str]) -> Optional[int]:
        """
        Find the starting position of a section using multiple patterns.

        Args:
            text: The filing text content.
            patterns: List of regex patterns to match section header.

        Returns:
            Starting position of the section or None if not found.
        """
        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                # Return the position of the last match (avoid table of contents)
                # Usually the actual content comes after TOC
                if len(matches) > 1:
                    return matches[-1].start()
                return matches[0].start()
        return None

    def _find_section_end(self, text: str, start_pos: int, section_name: str) -> int:
        """
        Find the ending position of a section.

        Args:
            text: The filing text content.
            start_pos: Starting position of the current section.
            section_name: Name of the current section.

        Returns:
            Ending position of the section.
        """
        # Determine which patterns to use for finding the end
        # based on which section we're in
        search_text = text[start_pos + 100:]  # Skip past current section header

        # Find the next section
        earliest_pos = len(search_text)

        for pattern in self.NEXT_SECTION_PATTERNS:
            # Skip patterns that match the current section
            if section_name.lower() in pattern.lower():
                continue

            matches = list(re.finditer(pattern, search_text))
            for match in matches:
                # Only consider matches that are likely section headers
                # (typically followed by newlines or significant text)
                pos = match.start()
                if pos < earliest_pos and pos > 50:  # Ensure we captured some content
                    earliest_pos = pos
                    break

        return start_pos + 100 + earliest_pos

    def _extract_sections(self, filing_path: str) -> dict:
        """
        Extract key sections from a SEC 10-K filing.

        Args:
            filing_path: Path to the 10-K filing file.

        Returns:
            Dictionary mapping section names to their text content.

        Raises:
            MalformedFilingError: If the filing cannot be read.
            SectionExtractionError: If no sections can be extracted.
        """
        filepath = Path(filing_path)

        if not filepath.exists():
            raise MalformedFilingError(f"Filing not found: {filing_path}")

        # Read the filing content
        try:
            # Try different encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    with open(filepath, "r", encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise MalformedFilingError(f"Unable to decode filing: {filing_path}")

        except IOError as e:
            raise MalformedFilingError(f"Error reading filing {filing_path}: {e}") from e

        if not text or len(text) < 1000:
            raise MalformedFilingError(f"Filing appears to be empty or too short: {filing_path}")

        # Clean the text
        text = self._clean_text(text)

        # Extract each section
        sections = {}

        for section_name, patterns in self.SECTION_PATTERNS.items():
            try:
                start_pos = self._find_section_start(text, patterns)

                if start_pos is not None:
                    end_pos = self._find_section_end(text, start_pos, section_name)
                    section_text = text[start_pos:end_pos].strip()

                    # Validate section content
                    if len(section_text) > 100:  # Minimum content threshold
                        sections[section_name] = section_text
                        logger.debug(
                            f"Extracted {section_name}: {len(section_text)} characters"
                        )
                    else:
                        logger.warning(
                            f"Section {section_name} too short ({len(section_text)} chars), skipping"
                        )
                else:
                    logger.warning(f"Section not found: {section_name}")

            except Exception as e:
                logger.warning(f"Error extracting section {section_name}: {e}")
                continue

        if not sections:
            raise SectionExtractionError(
                f"No valid sections extracted from filing: {filing_path}"
            )

        logger.info(f"Extracted {len(sections)} sections from {filepath.name}")
        return sections

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text from SEC filings.

        Args:
            text: Raw text content.

        Returns:
            Cleaned text.
        """
        # Remove HTML/XML tags if present
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Restore paragraph breaks
        text = re.sub(r"(\. )([A-Z])", r".\n\n\2", text)

        # Remove page numbers and headers/footers patterns
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        text = re.sub(r"(?i)table\s*of\s*contents?", "", text)

        # Remove excessive special characters
        text = re.sub(r"[_]{3,}", "", text)
        text = re.sub(r"[-]{3,}", "", text)
        text = re.sub(r"[=]{3,}", "", text)

        return text.strip()

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks using LangChain's RecursiveCharacterTextSplitter.

        Args:
            text: Text content to chunk.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        chunks = self.text_splitter.split_text(text)
        return chunks

    def process_sec_filing(self, filing_path: str) -> list[Document]:
        """
        Process a SEC 10-K filing into LangChain Document objects.

        This method extracts key sections from the filing, chunks the text,
        and creates Document objects with comprehensive metadata.

        Args:
            filing_path: Path to the SEC 10-K filing file.

        Returns:
            List of LangChain Document objects.

        Raises:
            MalformedFilingError: If the filing cannot be parsed.
            SectionExtractionError: If no sections can be extracted.

        Example:
            >>> processor = DocumentProcessor()
            >>> docs = processor.process_sec_filing("data/raw/apple_10k_2024.txt")
            >>> print(docs[0].metadata)
            {'source': 'data/raw/apple_10k_2024.txt', 'section': 'Item 1: Business', ...}
        """
        filepath = Path(filing_path)
        logger.info(f"Processing filing: {filepath.name}")

        # Read file for metadata extraction
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
        except IOError as e:
            raise MalformedFilingError(f"Cannot read filing: {e}") from e

        # Extract metadata
        company = self._extract_company_name(full_text, filing_path)
        period = self._extract_period(full_text, filing_path)

        logger.info(f"Detected company: {company}, period: {period}")

        # Extract sections
        sections = self._extract_sections(filing_path)

        # Process each section into Document objects
        documents = []
        global_chunk_id = 0

        for section_name, section_text in sections.items():
            # Chunk the section text
            chunks = self._chunk_text(section_text)

            logger.debug(f"Section '{section_name}' split into {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                # Create Document with metadata
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": str(filepath),
                        "section": section_name,
                        "chunk_id": global_chunk_id,
                        "section_chunk_id": i,
                        "doc_type": "10-K",
                        "company": company,
                        "period": period,
                        "chunk_size": len(chunk),
                        "total_section_chunks": len(chunks),
                    },
                )
                documents.append(doc)
                global_chunk_id += 1

        logger.info(
            f"Created {len(documents)} Document objects from {len(sections)} sections"
        )
        return documents

    def process_directory(
        self,
        input_dir: Path,
        output_path: Path,
        file_patterns: Optional[list[str]] = None,
    ) -> list[Document]:
        """
        Process all filings in a directory and save results.

        Args:
            input_dir: Directory containing SEC filings.
            output_path: Path to save the processed documents.
            file_patterns: List of file patterns to match (default: common text formats).

        Returns:
            List of all processed Document objects.
        """
        input_dir = Path(input_dir)
        output_path = Path(output_path)

        if not input_dir.exists():
            logger.warning(f"Input directory does not exist: {input_dir}")
            return []

        # Default file patterns
        if file_patterns is None:
            file_patterns = ["*.txt", "*.htm", "*.html", "*.10k", "*.10-k"]

        # Find all matching files
        files = []
        for pattern in file_patterns:
            files.extend(input_dir.rglob(pattern))

        # Remove duplicates and sort
        files = sorted(set(files))

        if not files:
            logger.warning(f"No matching files found in {input_dir}")
            return []

        logger.info(f"Found {len(files)} files to process")

        # Process each file with progress tracking
        all_documents = []
        successful = 0
        failed = 0

        for filepath in tqdm(files, desc="Processing filings", unit="file"):
            try:
                documents = self.process_sec_filing(str(filepath))
                all_documents.extend(documents)
                successful += 1

            except MalformedFilingError as e:
                logger.error(f"Malformed filing {filepath.name}: {e}")
                failed += 1

            except SectionExtractionError as e:
                logger.error(f"Section extraction failed for {filepath.name}: {e}")
                failed += 1

            except Exception as e:
                logger.error(f"Unexpected error processing {filepath.name}: {e}")
                failed += 1

        # Summary
        logger.info(
            f"Processing complete: {successful} successful, {failed} failed, "
            f"{len(all_documents)} total documents"
        )

        # Save processed documents
        if all_documents:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                pickle.dump(all_documents, f)

            logger.info(f"Saved {len(all_documents)} documents to {output_path}")

        return all_documents


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process SEC 10-K filings into LangChain Document objects.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all filings in data/raw/
  python process_documents.py

  # Process a single filing
  python process_documents.py --file data/raw/apple_10k.txt

  # Custom chunk size
  python process_documents.py --chunk-size 500 --chunk-overlap 100

  # Custom input/output directories
  python process_documents.py --input-dir ./filings --output ./processed/docs.pkl
        """,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=Path("data/raw"),
        help="Input directory containing SEC filings (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed/documents.pkl"),
        help="Output path for processed documents (default: data/processed/documents.pkl)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Process a single filing file instead of a directory",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI."""
    args = parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Initialize processor
    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    try:
        if args.file:
            # Process single file
            if not args.file.exists():
                logger.error(f"File not found: {args.file}")
                return 1

            logger.info(f"Processing single file: {args.file}")
            documents = processor.process_sec_filing(str(args.file))

            # Save output
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "wb") as f:
                pickle.dump(documents, f)

            logger.info(f"Saved {len(documents)} documents to {args.output}")

        else:
            # Process directory
            documents = processor.process_directory(
                input_dir=args.input_dir,
                output_path=args.output,
            )

            if not documents:
                logger.warning("No documents were processed")
                return 0

        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total documents created: {len(documents)}")

        if documents:
            # Group by section
            sections = {}
            companies = set()
            for doc in documents:
                section = doc.metadata.get("section", "Unknown")
                sections[section] = sections.get(section, 0) + 1
                companies.add(doc.metadata.get("company", "Unknown"))

            print(f"\nCompanies processed: {len(companies)}")
            for company in sorted(companies):
                print(f"  - {company}")

            print(f"\nDocuments by section:")
            for section, count in sorted(sections.items()):
                print(f"  - {section}: {count}")

            print(f"\nOutput saved to: {args.output}")

        print("=" * 60)
        return 0

    except DocumentProcessorError as e:
        logger.error(f"Processing failed: {e}")
        return 1

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())

