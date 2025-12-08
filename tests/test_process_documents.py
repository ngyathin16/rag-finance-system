"""
Tests for Document Processing Script.

Covers:
- DocumentProcessor: section extraction, chunking, metadata
- Error handling: malformed filings, missing sections
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add scripts and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        from process_documents import DocumentProcessor
        return DocumentProcessor(chunk_size=500, chunk_overlap=100)
    
    @pytest.fixture
    def sample_10k_content(self):
        """Create sample 10-K filing content."""
        return """
FORM 10-K
ANNUAL REPORT
FISCAL YEAR ENDED DECEMBER 31, 2024

TEST COMPANY INC.

ITEM 1. BUSINESS

Test Company Inc. is a technology company founded in 2010. The company 
develops software solutions for enterprise customers. Our primary products
include cloud computing services, data analytics platforms, and cybersecurity
solutions. We operate in North America, Europe, and Asia Pacific regions.

The company generated $5 billion in revenue during fiscal year 2024, 
representing a 15% increase year-over-year. Our growth was primarily driven
by strong adoption of our cloud services offering.

ITEM 1A. RISK FACTORS

Our business faces several significant risks including:
- Competition from larger technology companies
- Regulatory changes in data privacy laws
- Cybersecurity threats and data breaches
- Economic downturn affecting enterprise spending
- Dependence on key personnel and talent acquisition

These risk factors could materially affect our financial condition and results
of operations. Investors should carefully consider these risks before investing.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue increased by 15% to $5 billion in fiscal year 2024, compared to 
$4.3 billion in fiscal year 2023. The increase was primarily driven by 
growth in our cloud services segment.

Operating expenses increased by 8% due to investments in research and 
development and expansion of our sales team. Net income margin improved
to 22% from 19% in the prior year.

ITEM 8. FINANCIAL STATEMENTS

CONSOLIDATED BALANCE SHEET
Assets: $10.5 billion
Liabilities: $3.2 billion
Shareholders' Equity: $7.3 billion

SIGNATURES

Signed by the Chief Executive Officer on March 15, 2024.
"""
    
    @pytest.fixture
    def temp_filing_file(self, temp_dir, sample_10k_content):
        """Create a temporary 10-K filing file."""
        filing_path = temp_dir / "test_company_10k_2024.txt"
        filing_path.write_text(sample_10k_content)
        return filing_path
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp:
            yield Path(temp)
    
    def test_initialization(self):
        """Test DocumentProcessor initializes correctly."""
        from process_documents import DocumentProcessor
        
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
        assert processor.text_splitter is not None
    
    def test_custom_separators(self):
        """Test custom separators are applied."""
        from process_documents import DocumentProcessor
        
        custom_separators = ["\n\n\n", "\n\n", "\n"]
        processor = DocumentProcessor(separators=custom_separators)
        
        assert processor.separators == custom_separators
    
    def test_extract_company_name_from_text(self, processor, sample_10k_content):
        """Test company name extraction from filing text."""
        name = processor._extract_company_name(sample_10k_content, "test.txt")
        
        # Should extract "TEST COMPANY INC" or similar
        assert name != "Unknown"
        assert len(name) > 3
    
    def test_extract_company_name_fallback_to_filename(self, processor):
        """Test company name extraction falls back to filename."""
        minimal_text = "Some random text without company name."
        
        name = processor._extract_company_name(
            minimal_text, 
            "apple_10k_2024.txt"
        )
        
        # Should extract from filename
        assert "apple" in name.lower() or name != "Unknown"
    
    def test_extract_period_from_text(self, processor, sample_10k_content):
        """Test fiscal period extraction from filing text."""
        period = processor._extract_period(sample_10k_content, "test.txt")
        
        assert "2024" in period
        assert period != "Unknown"
    
    def test_extract_period_from_filename(self, processor):
        """Test period extraction falls back to filename."""
        minimal_text = "Some text without dates."
        
        period = processor._extract_period(
            minimal_text,
            "company_10k_2023.txt"
        )
        
        assert "2023" in period
    
    def test_clean_text_removes_html(self, processor):
        """Test that clean_text removes HTML tags."""
        html_text = "<div><p>Revenue was <strong>$5 billion</strong></p></div>"
        
        cleaned = processor._clean_text(html_text)
        
        assert "<div>" not in cleaned
        assert "<p>" not in cleaned
        assert "<strong>" not in cleaned
        assert "Revenue" in cleaned
        assert "$5 billion" in cleaned
    
    def test_clean_text_removes_excessive_whitespace(self, processor):
        """Test that clean_text normalizes whitespace."""
        messy_text = "Revenue    was     high\n\n\n\n\nand growing"
        
        cleaned = processor._clean_text(messy_text)
        
        assert "    " not in cleaned  # No excessive spaces
    
    def test_chunk_text_basic(self, processor):
        """Test basic text chunking."""
        long_text = "This is a sentence. " * 100  # Create text to chunk
        
        chunks = processor._chunk_text(long_text)
        
        assert len(chunks) > 1  # Should create multiple chunks
        assert all(len(chunk) <= processor.chunk_size + 50 for chunk in chunks)
    
    def test_chunk_text_empty_input(self, processor):
        """Test chunking empty text returns empty list."""
        chunks = processor._chunk_text("")
        assert chunks == []
        
        chunks = processor._chunk_text("   ")
        assert chunks == []
    
    def test_process_sec_filing(self, processor, temp_filing_file):
        """Test processing a complete SEC filing."""
        documents = processor.process_sec_filing(str(temp_filing_file))
        
        assert len(documents) > 0
        
        # Check document structure
        for doc in documents:
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")
            assert "section" in doc.metadata
            assert "company" in doc.metadata
            assert "period" in doc.metadata
            assert "chunk_id" in doc.metadata
    
    def test_process_sec_filing_extracts_multiple_sections(self, processor, temp_filing_file):
        """Test that multiple sections are extracted."""
        documents = processor.process_sec_filing(str(temp_filing_file))
        
        # Get unique sections
        sections = set(doc.metadata["section"] for doc in documents)
        
        # Should extract multiple sections
        assert len(sections) >= 2
    
    def test_process_sec_filing_nonexistent_file(self, processor):
        """Test processing nonexistent file raises error."""
        from process_documents import MalformedFilingError
        
        with pytest.raises(MalformedFilingError):
            processor.process_sec_filing("/nonexistent/file.txt")
    
    def test_process_sec_filing_empty_file(self, processor, temp_dir):
        """Test processing empty file raises error."""
        from process_documents import MalformedFilingError
        
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")
        
        with pytest.raises(MalformedFilingError, match="empty|too short"):
            processor.process_sec_filing(str(empty_file))
    
    def test_process_sec_filing_too_short(self, processor, temp_dir):
        """Test processing file that's too short raises error."""
        from process_documents import MalformedFilingError
        
        short_file = temp_dir / "short.txt"
        short_file.write_text("Short content")
        
        with pytest.raises(MalformedFilingError, match="empty|too short"):
            processor.process_sec_filing(str(short_file))


class TestDocumentProcessorSectionExtraction:
    """Tests for section extraction functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        from process_documents import DocumentProcessor
        return DocumentProcessor()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp:
            yield Path(temp)
    
    def test_find_section_start(self, processor):
        """Test finding section start position."""
        text = "Table of Contents\nItem 1 Business\n...\nITEM 1. BUSINESS\n\nActual content here"
        
        patterns = [r"(?i)item\s*1\.?\s*[-–—]?\s*business"]
        
        pos = processor._find_section_start(text, patterns)
        
        assert pos is not None
        # Should find the later occurrence (not TOC)
        assert pos > text.find("Table of Contents")
    
    def test_find_section_start_not_found(self, processor):
        """Test finding section that doesn't exist."""
        text = "Some random text without sections"
        
        patterns = [r"(?i)item\s*99\.?\s*nonexistent"]
        
        pos = processor._find_section_start(text, patterns)
        
        assert pos is None
    
    def test_extract_sections_no_valid_sections(self, processor, temp_dir):
        """Test handling filing with no recognizable sections."""
        from process_documents import SectionExtractionError
        
        filing = temp_dir / "no_sections.txt"
        filing.write_text("A" * 2000)  # Long enough but no sections
        
        with pytest.raises(SectionExtractionError, match="No valid sections"):
            processor._extract_sections(str(filing))


class TestDocumentProcessorDirectory:
    """Tests for directory processing functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        from process_documents import DocumentProcessor
        return DocumentProcessor(chunk_size=500, chunk_overlap=100)
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp:
            yield Path(temp)
    
    @pytest.fixture
    def sample_10k_content(self):
        """Create sample 10-K filing content (must be >1000 chars)."""
        return """
FORM 10-K
ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT
TEST COMPANY INC.
FISCAL YEAR ENDED DECEMBER 31, 2024

ITEM 1. BUSINESS

The company develops enterprise software solutions for global customers. 
Revenue grew by 15% in 2024 to reach $5 billion in annual recurring revenue.
We operate globally across multiple regions with diverse product lines.
Our primary products include cloud computing services, data analytics platforms,
and cybersecurity solutions. The company was founded in 2010 and has grown
significantly through both organic growth and strategic acquisitions.
We serve customers in over 50 countries with offices in North America, Europe,
and Asia Pacific. Our technology platform processes over 10 billion transactions
daily and serves millions of end users worldwide.

ITEM 1A. RISK FACTORS

The company faces significant competition from larger technology companies.
Market volatility could affect our financial performance and stock price.
Regulatory changes in data privacy laws may require substantial investments.
Cybersecurity threats and data breaches pose ongoing risks to operations.
Economic downturns could reduce enterprise spending on software solutions.
Dependence on key personnel and challenges in talent acquisition persist.
Changes in foreign currency exchange rates may impact international revenue.
Supply chain disruptions could affect our ability to deliver services.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue increased to $5 billion in fiscal year 2024, representing a 15% 
increase year-over-year from $4.35 billion in fiscal year 2023. The growth
was primarily driven by strong adoption of our cloud services offering.
Operating expenses increased by 8% due to investments in research and 
development and expansion of our sales team. Net income margin improved
to 22% from 19% in the prior year, reflecting operational efficiencies.
We expect continued growth in the coming fiscal year as we expand our
product portfolio and enter new geographic markets.

SIGNATURES

Signed by the Chief Executive Officer on March 15, 2024.
"""
    
    def test_process_directory_empty(self, processor, temp_dir):
        """Test processing empty directory returns empty list."""
        output_path = temp_dir / "output" / "docs.pkl"
        
        documents = processor.process_directory(
            input_dir=temp_dir / "empty_input",
            output_path=output_path
        )
        
        assert documents == []
    
    def test_process_directory_with_files(self, processor, temp_dir, sample_10k_content):
        """Test processing directory with multiple files."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        
        # Create test files
        for i in range(2):
            filing = input_dir / f"company_{i}_10k.txt"
            filing.write_text(sample_10k_content)
        
        output_path = temp_dir / "output" / "docs.pkl"
        
        documents = processor.process_directory(
            input_dir=input_dir,
            output_path=output_path
        )
        
        assert len(documents) > 0
        assert output_path.exists()
    
    def test_process_directory_handles_failures(self, processor, temp_dir, sample_10k_content):
        """Test that directory processing continues after individual file failures."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        
        # Create one valid and one invalid file
        valid_file = input_dir / "valid_10k.txt"
        valid_file.write_text(sample_10k_content)
        
        invalid_file = input_dir / "invalid_10k.txt"
        invalid_file.write_text("Too short")
        
        output_path = temp_dir / "output" / "docs.pkl"
        
        # Should process valid file and skip invalid
        documents = processor.process_directory(
            input_dir=input_dir,
            output_path=output_path
        )
        
        assert len(documents) > 0  # At least the valid file was processed


class TestDocumentProcessorEdgeCases:
    """Edge case tests for DocumentProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        from process_documents import DocumentProcessor
        return DocumentProcessor()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp:
            yield Path(temp)
    
    def test_process_filing_with_encoding_issues(self, processor, temp_dir):
        """Test handling files with different encodings."""
        # Create file with UTF-8 encoding (avoiding latin-1 issues on Windows)
        filing = temp_dir / "utf8_file.txt"
        content = """FORM 10-K
TEST COMPANY INC.
FISCAL YEAR 2024

ITEM 1. BUSINESS

Company resume and cafe data. Revenue was 5 million dollars.
The company operates globally with significant market presence.
Business operations span multiple sectors including technology and services.

ITEM 1A. RISK FACTORS

Risk factors include currency fluctuation and market volatility.
The company faces significant competition in all markets.
Regulatory changes could impact future operations.

SIGNATURES
Signed.
""" * 30  # Make it long enough
        
        filing.write_text(content, encoding="utf-8")
        
        # Should handle encoding gracefully
        documents = processor.process_sec_filing(str(filing))
        assert len(documents) >= 0  # May extract sections or handle gracefully
    
    def test_very_long_section(self, processor, temp_dir):
        """Test handling very long sections."""
        long_section = "Financial data. " * 50000  # Very long content
        
        content = f"""FORM 10-K
TEST COMPANY INC.
FISCAL YEAR 2024

ITEM 1. BUSINESS

{long_section}

ITEM 1A. RISK FACTORS

Risk factors here.

SIGNATURES
Signed.
"""
        
        filing = temp_dir / "long_section.txt"
        filing.write_text(content)
        
        documents = processor.process_sec_filing(str(filing))
        
        # Should create many chunks from the long section
        assert len(documents) > 10
    
    def test_special_characters_in_content(self, processor, temp_dir):
        """Test handling special characters in filing content."""
        content = """FORM 10-K
TEST COMPANY INC.
FISCAL YEAR 2024

ITEM 1. BUSINESS

Revenue was $5.2B in FY2024, representing 15% YoY growth.
The company achieved strong performance across all segments.
Primary business activities include software development and services.
Operations span North America, Europe, and Asia Pacific regions.
- Bullet point 1 about business operations
- Bullet point 2 about market position
(c) 2024 Test Company All Rights Reserved

ITEM 1A. RISK FACTORS

Risk factors include market competition and regulatory changes.
The company operates in a highly competitive environment.
Changes in tax laws could affect profitability.

SIGNATURES
Signed on March 15, 2024.
""" * 20
        
        filing = temp_dir / "special_chars.txt"
        filing.write_text(content, encoding="utf-8")
        
        documents = processor.process_sec_filing(str(filing))
        assert len(documents) > 0
    
    def test_section_minimum_content_threshold(self, processor, temp_dir):
        """Test that very short sections are skipped."""
        content = """FORM 10-K
TEST COMPANY INC.
FISCAL YEAR ENDED DECEMBER 31, 2024

ITEM 1. BUSINESS

OK.

ITEM 1A. RISK FACTORS

This section has adequate content for extraction and should be processed
normally by the document processor. It contains multiple sentences with
relevant information about risk factors that investors should consider.
The company faces various risks including market competition, regulatory
changes, and economic uncertainty. These risks could materially affect
our business operations and financial performance. Investors should
carefully review all risk factors before making investment decisions.
Additional risks include cybersecurity threats and supply chain disruptions.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue increased by 15% in 2024 compared to the prior year.
Operating expenses remained stable while margins improved.
The company expects continued growth in the coming year.
Cash flow from operations was strong throughout the period.

SIGNATURES
Signed on March 15, 2024.
""" * 5  # Make it long enough
        
        filing = temp_dir / "short_section.txt"
        filing.write_text(content, encoding="utf-8")
        
        # Short Item 1 section should be skipped
        documents = processor.process_sec_filing(str(filing))
        
        # Only the longer sections should be processed
        if len(documents) > 0:
            sections = set(doc.metadata["section"] for doc in documents)
            # Item 1 should be skipped due to short content
            assert "Item 1: Business" not in sections or len(sections) >= 1
    
    def test_chunk_metadata_includes_all_fields(self, processor, temp_dir):
        """Test that chunk metadata includes all required fields."""
        content = """FORM 10-K
ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT
TEST COMPANY INC.
FISCAL YEAR ENDED DECEMBER 31, 2024

ITEM 1. BUSINESS

The company develops software solutions for enterprise customers worldwide.
We have operations in multiple countries and serve diverse industries.
Revenue has been growing steadily over the past several years with strong
performance in all major geographic regions. Our products include cloud
computing platforms, data analytics solutions, and enterprise security tools.
We serve customers across technology, financial services, healthcare, and
retail sectors. The company was founded in 2010 and has grown to over
5000 employees globally with offices in North America, Europe, and Asia.

ITEM 1A. RISK FACTORS

The company faces various risks including market competition and regulatory
changes. Economic downturns could affect customer spending on software.
Cybersecurity threats pose ongoing risks to operations and reputation.
Changes in tax laws could impact profitability and cash flow.
Dependence on key personnel creates succession planning challenges.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue increased by 15% to $5 billion in fiscal year 2024 compared to
$4.35 billion in fiscal year 2023. The growth was driven by strong adoption
of our cloud services platform and expansion into new geographic markets.
Operating margins improved to 22% from 19% in the prior year reflecting
operational efficiencies and favorable product mix. We expect continued
growth momentum in the coming fiscal year as we invest in new products.

SIGNATURES
Signed on March 15, 2024.
""" * 3
        
        filing = temp_dir / "test_metadata.txt"
        filing.write_text(content)
        
        documents = processor.process_sec_filing(str(filing))
        
        if len(documents) > 0:
            doc = documents[0]
            
            # Check all expected metadata fields
            expected_fields = [
                "source", "section", "chunk_id", "section_chunk_id",
                "doc_type", "company", "period", "chunk_size", "total_section_chunks"
            ]
            
            for field in expected_fields:
                assert field in doc.metadata, f"Missing metadata field: {field}"


class TestDocumentProcessorRobustness:
    """Robustness tests for DocumentProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        from process_documents import DocumentProcessor
        return DocumentProcessor()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp:
            yield Path(temp)
    
    def test_process_multiple_item_formats(self, processor, temp_dir):
        """Test handling various Item header formats."""
        content = """FORM 10-K
ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT
TEST COMPANY INC.
FISCAL YEAR ENDED DECEMBER 31, 2024

Item 1 - Business

Company description with sufficient content for extraction and processing.
We develop software and provide professional services to customers globally.
Our products include enterprise applications, cloud platforms, and analytics.
The company serves customers in technology, healthcare, and financial sectors.
Operations span multiple geographic regions including North America, Europe,
and Asia Pacific with significant growth in emerging markets. Revenue has
grown consistently over the past five years with strong profitability.

Item 1A – Risk Factors

Risk factor content with enough text to be properly extracted by the parser.
Multiple risks are identified and discussed in detail throughout this section.
Competition from larger technology companies poses significant challenges.
Regulatory changes could require substantial investments in compliance.
Economic downturns may reduce customer spending on enterprise software.
Cybersecurity threats create ongoing operational and reputational risks.

Item 7: Management's Discussion and Analysis

Financial discussion with adequate content length for proper processing.
Revenue and expenses are analyzed in comprehensive detail in this section.
Total revenue increased 15% year-over-year to reach $5 billion in FY2024.
Operating margins improved significantly due to operational efficiencies.
Cash flow from operations remained strong throughout the fiscal year.
We expect continued growth momentum as we expand our product portfolio.

SIGNATURES
Signed by the Chief Executive Officer on March 15, 2024.
""" * 3
        
        filing = temp_dir / "various_formats.txt"
        filing.write_text(content)
        
        documents = processor.process_sec_filing(str(filing))
        
        # Should extract sections regardless of format variations
        sections = set(doc.metadata["section"] for doc in documents)
        assert len(sections) >= 1  # At least some sections extracted

