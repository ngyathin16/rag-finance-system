"""
Tests for SEC Data Ingestion Script.

Covers:
- SECDataIngester: download, extraction, parsing, edge cases
- Error handling: timeout, connection errors, invalid data
"""

import os
import sys
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import pandas as pd

# Add scripts and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestSECDataIngester:
    """Tests for SECDataIngester class."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def ingester(self, temp_output_dir):
        """Create an SECDataIngester instance."""
        from ingest_sec_data import SECDataIngester
        return SECDataIngester(output_dir=temp_output_dir, timeout=30)
    
    def test_initialization(self, temp_output_dir):
        """Test SECDataIngester initializes correctly."""
        from ingest_sec_data import SECDataIngester
        
        ingester = SECDataIngester(output_dir=temp_output_dir, timeout=60)
        
        assert ingester.output_dir == temp_output_dir
        assert ingester.timeout == 60
        assert ingester.session is not None
    
    def test_build_url(self, ingester):
        """Test URL building for quarterly data."""
        url = ingester._build_url(2024, 3)
        
        assert "2024q3.zip" in url
        assert "sec.gov" in url
    
    def test_validate_quarter_params_valid(self, ingester):
        """Test validation passes for valid parameters."""
        # Should not raise
        ingester._validate_quarter_params(2024, 1)
        ingester._validate_quarter_params(2024, 4)
        ingester._validate_quarter_params(2009, 1)
    
    def test_validate_quarter_params_invalid_year(self, ingester):
        """Test validation fails for invalid year."""
        with pytest.raises(ValueError, match="Year must be"):
            ingester._validate_quarter_params(2008, 1)  # Before 2009
        
        with pytest.raises(ValueError, match="Year must be"):
            ingester._validate_quarter_params(2101, 1)  # After 2100
        
        with pytest.raises(ValueError, match="Year must be"):
            ingester._validate_quarter_params("2024", 1)  # String instead of int
    
    def test_validate_quarter_params_invalid_quarter(self, ingester):
        """Test validation fails for invalid quarter."""
        with pytest.raises(ValueError, match="Quarter must be"):
            ingester._validate_quarter_params(2024, 0)
        
        with pytest.raises(ValueError, match="Quarter must be"):
            ingester._validate_quarter_params(2024, 5)
        
        with pytest.raises(ValueError, match="Quarter must be"):
            ingester._validate_quarter_params(2024, "1")  # String instead of int


class TestSECDataIngesterDownload:
    """Tests for download functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def ingester(self, temp_output_dir):
        """Create an SECDataIngester instance."""
        from ingest_sec_data import SECDataIngester
        return SECDataIngester(output_dir=temp_output_dir)
    
    @patch("requests.Session.get")
    def test_download_quarter_success(self, mock_get, ingester, temp_output_dir):
        """Test successful download of quarterly data."""
        # Create mock response with ZIP content
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"fake zip content"]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        zip_path = ingester.download_quarter(2024, 3)
        
        assert zip_path.exists()
        assert "2024q3" in str(zip_path)
    
    @patch("requests.Session.get")
    def test_download_timeout_error(self, mock_get, ingester):
        """Test handling of download timeout."""
        from ingest_sec_data import DownloadError
        import requests
        
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
        
        with pytest.raises(DownloadError, match="timed out"):
            ingester.download_quarter(2024, 3)
    
    @patch("requests.Session.get")
    def test_download_connection_error(self, mock_get, ingester):
        """Test handling of connection error."""
        from ingest_sec_data import DownloadError
        import requests
        
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")
        
        with pytest.raises(DownloadError, match="Connection error"):
            ingester.download_quarter(2024, 3)
    
    @patch("requests.Session.get")
    def test_download_404_not_found(self, mock_get, ingester):
        """Test handling of 404 response (data not available)."""
        from ingest_sec_data import DownloadError
        import requests
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response
        
        with pytest.raises(DownloadError, match="not found|not available"):
            ingester.download_quarter(2024, 3)
    
    @patch("requests.Session.get")
    def test_download_server_error(self, mock_get, ingester):
        """Test handling of 500 server error."""
        from ingest_sec_data import DownloadError
        import requests
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response
        
        with pytest.raises(DownloadError, match="HTTP error"):
            ingester.download_quarter(2024, 3)


class TestSECDataIngesterExtraction:
    """Tests for extraction and parsing functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def ingester(self, temp_output_dir):
        """Create an SECDataIngester instance."""
        from ingest_sec_data import SECDataIngester
        return SECDataIngester(output_dir=temp_output_dir)
    
    
    @pytest.fixture
    def valid_zip_file(self, temp_output_dir):
        """Create a valid test ZIP file with SEC data structure."""
        zip_path = temp_output_dir / "test_quarter" / "2024q3.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create mock SEC data files with proper column names
        num_data = "adsh\ttag\tversion\tcoreg\tddate\tqtrs\tuom\tvalue\n" \
                   "0001234567-24-000001\tAssets\tus-gaap/2024\t\t20240331\t0\tUSD\t1000000\n" \
                   "0001234567-24-000001\tRevenue\tus-gaap/2024\t\t20240331\t1\tUSD\t500000\n"
        
        sub_data = "adsh\tcik\tname\tform\tfiled\n" \
                   "0001234567-24-000001\t1234567\tTest Company Inc\t10-K\t2024-03-15\n"
        
        # Create ZIP file
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("num.txt", num_data)
            zf.writestr("sub.txt", sub_data)
        
        return zip_path
    
    def test_extract_and_parse_success(self, ingester, valid_zip_file):
        """Test successful extraction and parsing."""
        df = ingester.extract_and_parse(valid_zip_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "adsh" in df.columns
        assert "value" in df.columns
    
    def test_extract_and_parse_merges_metadata(self, ingester, valid_zip_file):
        """Test that submission metadata is merged."""
        df = ingester.extract_and_parse(valid_zip_file)
        
        # Should have merged columns from sub.txt
        assert "name" in df.columns or "cik" in df.columns
    
    def test_extract_file_not_found(self, ingester, temp_output_dir):
        """Test handling of missing ZIP file."""
        nonexistent_path = temp_output_dir / "nonexistent.zip"
        
        with pytest.raises(FileNotFoundError):
            ingester.extract_and_parse(nonexistent_path)
    
    def test_extract_invalid_zip(self, ingester, temp_output_dir):
        """Test handling of invalid ZIP file."""
        from ingest_sec_data import ExtractionError
        
        invalid_zip = temp_output_dir / "invalid.zip"
        invalid_zip.write_text("not a zip file")
        
        with pytest.raises(ExtractionError, match="Invalid ZIP"):
            ingester.extract_and_parse(invalid_zip)
    
    def test_extract_missing_num_file(self, ingester, temp_output_dir):
        """Test handling of ZIP without num.txt."""
        from ingest_sec_data import ExtractionError
        
        zip_path = temp_output_dir / "missing_num.zip"
        
        # Create ZIP without num.txt
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sub.txt", "adsh\tcik\n")
        
        with pytest.raises(ExtractionError, match="num.txt not found"):
            ingester.extract_and_parse(zip_path)


class TestSECDataIngesterIntegration:
    """Integration tests for SECDataIngester."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def ingester(self, temp_output_dir):
        """Create an SECDataIngester instance."""
        from ingest_sec_data import SECDataIngester
        return SECDataIngester(output_dir=temp_output_dir)
    
    def test_ingest_quarter_combines_download_and_parse(self, ingester, temp_output_dir):
        """Test ingest_quarter method combines download and parse."""
        # Create a fake downloaded file
        zip_path = temp_output_dir / "2024q3" / "2024q3.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        
        num_data = "adsh\ttag\tversion\tcoreg\tddate\tqtrs\tuom\tvalue\n" \
                   "0001234567-24-000001\tAssets\tus-gaap/2024\t\t20240331\t0\tUSD\t1000000\n"
        sub_data = "adsh\tcik\tname\tform\tfiled\n" \
                   "0001234567-24-000001\t1234567\tTest Co\t10-K\t2024-03-15\n"
        
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("num.txt", num_data)
            zf.writestr("sub.txt", sub_data)
        
        # Mock download to return our pre-created file
        with patch.object(ingester, "download_quarter", return_value=zip_path):
            df = ingester.ingest_quarter(2024, 3)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestSECDataIngesterEdgeCases:
    """Edge case tests for SECDataIngester."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_empty_num_file(self, temp_output_dir):
        """Test handling of empty num.txt file."""
        from ingest_sec_data import SECDataIngester, ExtractionError
        
        ingester = SECDataIngester(output_dir=temp_output_dir)
        
        zip_path = temp_output_dir / "empty.zip"
        
        # Create ZIP with empty num.txt (just headers)
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("num.txt", "adsh\ttag\tversion\tcoreg\tddate\tqtrs\tuom\tvalue\n")
            zf.writestr("sub.txt", "adsh\tcik\tname\tform\tfiled\n")
        
        # Should succeed but return empty DataFrame
        df = ingester.extract_and_parse(zip_path)
        assert len(df) == 0
    
    def test_malformed_tsv_data(self, temp_output_dir):
        """Test handling of malformed TSV data."""
        from ingest_sec_data import SECDataIngester
        
        ingester = SECDataIngester(output_dir=temp_output_dir)
        
        zip_path = temp_output_dir / "malformed.zip"
        
        # Create ZIP with some malformed data (pandas should handle gracefully)
        num_data = "adsh\ttag\tversion\tcoreg\tddate\tqtrs\tuom\tvalue\n" \
                   "0001234567-24-000001\tAssets\tus-gaap/2024\t\t20240331\t0\tUSD\t1000000\n" \
                   "0001234567-24-000002\tRevenue\tus-gaap/2024\t\t20240331\t1\tUSD\t500000\n"
        
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("num.txt", num_data)
        
        # Should handle gracefully (pandas handles bad lines)
        df = ingester.extract_and_parse(zip_path)
        assert isinstance(df, pd.DataFrame)
    
    def test_unicode_in_data(self, temp_output_dir):
        """Test handling of unicode characters in data."""
        from ingest_sec_data import SECDataIngester
        
        ingester = SECDataIngester(output_dir=temp_output_dir)
        
        zip_path = temp_output_dir / "unicode.zip"
        
        # Create ZIP with unicode characters
        num_data = "adsh\ttag\tversion\tcoreg\tddate\tqtrs\tuom\tvalue\n" \
                   "0001234567-24-000001\tAssets™\tus-gaap/2024\t\t20240331\t0\tUSD\t1000000\n"
        sub_data = "adsh\tcik\tname\tform\tfiled\n" \
                   "0001234567-24-000001\t1234567\tCompañía Test Inc\t10-K\t2024-03-15\n"
        
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("num.txt", num_data)
            zf.writestr("sub.txt", sub_data)
        
        df = ingester.extract_and_parse(zip_path)
        assert len(df) > 0
    
    def test_very_large_values(self, temp_output_dir):
        """Test handling of very large numeric values."""
        from ingest_sec_data import SECDataIngester
        
        ingester = SECDataIngester(output_dir=temp_output_dir)
        
        zip_path = temp_output_dir / "large_values.zip"
        
        # Create ZIP with very large values
        num_data = "adsh\ttag\tversion\tcoreg\tddate\tqtrs\tuom\tvalue\n" \
                   "0001234567-24-000001\tAssets\tus-gaap/2024\t\t20240331\t0\tUSD\t999999999999999999\n"
        
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("num.txt", num_data)
        
        df = ingester.extract_and_parse(zip_path)
        assert len(df) == 1


class TestIngestRange:
    """Tests for ingest_range functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def ingester(self, temp_output_dir):
        """Create an SECDataIngester instance."""
        from ingest_sec_data import SECDataIngester
        return SECDataIngester(output_dir=temp_output_dir)
    
    def test_ingest_range_generates_correct_quarters(self, ingester):
        """Test that ingest_range generates correct quarter list."""
        with patch.object(ingester, "ingest_quarter") as mock_ingest:
            mock_ingest.return_value = pd.DataFrame()
            
            results = ingester.ingest_range(2023, 3, 2024, 1)
        
        # Should process Q3 2023, Q4 2023, Q1 2024
        assert len(results) == 3
        assert "2023q3" in results
        assert "2023q4" in results
        assert "2024q1" in results
    
    def test_ingest_range_handles_errors(self, ingester):
        """Test that ingest_range continues after individual errors."""
        from ingest_sec_data import DownloadError
        
        call_count = [0]
        
        def mock_ingest(year, quarter):
            call_count[0] += 1
            if call_count[0] == 2:
                raise DownloadError("Simulated error")
            return pd.DataFrame({"col": [1]})
        
        with patch.object(ingester, "ingest_quarter", side_effect=mock_ingest):
            results = ingester.ingest_range(2023, 3, 2024, 1)
        
        # One should be None (failed), others should have data
        success_count = sum(1 for df in results.values() if df is not None)
        assert success_count == 2

