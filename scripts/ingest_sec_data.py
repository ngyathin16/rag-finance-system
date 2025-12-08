#!/usr/bin/env python3
"""
SEC EDGAR Financial Statement Data Ingestion Script.

This script downloads and processes financial statement data sets from
the SEC EDGAR database for use in the RAG Finance System.

SEC Data Source: https://www.sec.gov/files/dera/data/financial-statement-data-sets
"""

import argparse
import logging
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
SEC_BASE_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
USER_AGENT = "MyRAGSystem/1.0 contact@example.com"
DEFAULT_TIMEOUT = 60  # seconds
CHUNK_SIZE = 8192  # bytes for streaming download


class SECDataIngesterError(Exception):
    """Base exception for SEC data ingestion errors."""

    pass


class DownloadError(SECDataIngesterError):
    """Exception raised when download fails."""

    pass


class ExtractionError(SECDataIngesterError):
    """Exception raised when extraction fails."""

    pass


class SECDataIngester:
    """
    Ingests SEC EDGAR financial statement data sets.

    This class provides methods to download quarterly financial statement
    data from SEC EDGAR and parse it into pandas DataFrames.

    Attributes:
        output_dir: Base directory for storing downloaded and extracted data.
        session: Requests session with configured headers.

    Example:
        >>> ingester = SECDataIngester(output_dir=Path("data/raw"))
        >>> zip_path = ingester.download_quarter(2024, 3)
        >>> df = ingester.extract_and_parse(zip_path)
    """

    def __init__(
        self,
        output_dir: Path = Path("data/raw"),
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize the SEC Data Ingester.

        Args:
            output_dir: Base directory for storing downloaded data.
            timeout: Request timeout in seconds.
        """
        self.output_dir = Path(output_dir)
        self.timeout = timeout

        # Create requests session with required headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/zip, application/octet-stream, */*",
            }
        )

        logger.info(f"Initialized SECDataIngester with output_dir: {self.output_dir}")

    def _build_url(self, year: int, quarter: int) -> str:
        """
        Build the download URL for a specific quarter.

        Args:
            year: The year (e.g., 2024).
            quarter: The quarter (1-4).

        Returns:
            The full URL to the quarterly data ZIP file.
        """
        # SEC uses format: {year}q{quarter}.zip
        filename = f"{year}q{quarter}.zip"
        return f"{SEC_BASE_URL}/{filename}"

    def _validate_quarter_params(self, year: int, quarter: int) -> None:
        """
        Validate year and quarter parameters.

        Args:
            year: The year to validate.
            quarter: The quarter to validate.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not isinstance(year, int) or year < 2009 or year > 2100:
            raise ValueError(f"Year must be an integer between 2009 and 2100, got: {year}")

        if not isinstance(quarter, int) or quarter < 1 or quarter > 4:
            raise ValueError(f"Quarter must be an integer between 1 and 4, got: {quarter}")

    def download_quarter(self, year: int, quarter: int) -> Path:
        """
        Download SEC financial statement data for a specific quarter.

        Downloads the quarterly data ZIP file from SEC EDGAR and saves it
        to the configured output directory.

        Args:
            year: The year (e.g., 2024).
            quarter: The quarter (1-4).

        Returns:
            Path to the downloaded ZIP file.

        Raises:
            ValueError: If year or quarter parameters are invalid.
            DownloadError: If the download fails due to network issues.

        Example:
            >>> ingester = SECDataIngester()
            >>> zip_path = ingester.download_quarter(2024, 3)
            >>> print(zip_path)
            data/raw/2024q3/2024q3.zip
        """
        self._validate_quarter_params(year, quarter)

        url = self._build_url(year, quarter)
        quarter_dir = self.output_dir / f"{year}q{quarter}"
        quarter_dir.mkdir(parents=True, exist_ok=True)

        zip_filename = f"{year}q{quarter}.zip"
        zip_path = quarter_dir / zip_filename

        logger.info(f"Downloading SEC data from: {url}")

        try:
            # Stream the download to handle large files
            response = self.session.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()

            # Get total file size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with open(zip_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {zip_filename}",
                    disable=total_size == 0,
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

            logger.info(f"Successfully downloaded: {zip_path}")
            return zip_path

        except requests.exceptions.Timeout as e:
            logger.error(f"Download timed out after {self.timeout}s: {url}")
            raise DownloadError(f"Download timed out: {e}") from e

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error while downloading: {url}")
            raise DownloadError(f"Connection error: {e}") from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "unknown"
            logger.error(f"HTTP error {status_code} while downloading: {url}")
            if status_code == 404:
                raise DownloadError(
                    f"Data not found for {year}Q{quarter}. "
                    "The quarter may not be available yet."
                ) from e
            raise DownloadError(f"HTTP error {status_code}: {e}") from e

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise DownloadError(f"Download failed: {e}") from e

    def extract_and_parse(self, zip_path: Path) -> pd.DataFrame:
        """
        Extract and parse SEC financial statement data from a ZIP file.

        Extracts all TSV/TXT files from the ZIP archive and combines the
        numeric data (num.txt) into a pandas DataFrame.

        Args:
            zip_path: Path to the downloaded ZIP file.

        Returns:
            DataFrame containing the parsed financial statement data.

        Raises:
            ExtractionError: If extraction or parsing fails.
            FileNotFoundError: If the ZIP file doesn't exist.

        Example:
            >>> ingester = SECDataIngester()
            >>> df = ingester.extract_and_parse(Path("data/raw/2024q3/2024q3.zip"))
            >>> print(df.columns.tolist())
            ['adsh', 'tag', 'version', 'coreg', 'ddate', 'qtrs', 'uom', 'value', ...]
        """
        zip_path = Path(zip_path)

        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        extract_dir = zip_path.parent
        logger.info(f"Extracting {zip_path} to {extract_dir}")

        try:
            # Extract all files from the ZIP
            with zipfile.ZipFile(zip_path, "r") as zf:
                file_list = zf.namelist()
                logger.info(f"Found {len(file_list)} files in archive: {file_list}")

                # Extract with progress bar
                for file_name in tqdm(file_list, desc="Extracting files"):
                    zf.extract(file_name, extract_dir)

            logger.info(f"Successfully extracted files to: {extract_dir}")

            # Parse the main data files
            dataframes = {}
            data_files = {
                "sub": "sub.txt",  # Submission data
                "num": "num.txt",  # Numeric data
                "tag": "tag.txt",  # Tag definitions
                "pre": "pre.txt",  # Presentation data
            }

            for name, filename in data_files.items():
                file_path = extract_dir / filename
                if file_path.exists():
                    logger.info(f"Parsing {filename}...")
                    try:
                        # SEC files are tab-separated
                        df = pd.read_csv(
                            file_path,
                            sep="\t",
                            low_memory=False,
                            encoding="utf-8",
                            on_bad_lines="warn",
                        )
                        dataframes[name] = df
                        logger.info(f"  Loaded {len(df):,} rows from {filename}")
                    except Exception as e:
                        logger.warning(f"  Failed to parse {filename}: {e}")
                else:
                    logger.warning(f"  File not found: {filename}")

            # Return the numeric data as the primary DataFrame
            # This contains the actual financial values
            if "num" in dataframes:
                num_df = dataframes["num"]

                # Add submission metadata if available
                if "sub" in dataframes:
                    logger.info("Merging submission metadata with numeric data...")
                    sub_df = dataframes["sub"][["adsh", "cik", "name", "form", "filed"]]
                    num_df = num_df.merge(sub_df, on="adsh", how="left")
                    logger.info(f"  Merged DataFrame has {len(num_df):,} rows")

                # Save the processed DataFrame
                output_path = extract_dir / "financial_data.parquet"
                num_df.to_parquet(output_path, index=False)
                logger.info(f"Saved processed data to: {output_path}")

                return num_df

            else:
                raise ExtractionError("num.txt not found in the ZIP archive")

        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file: {zip_path}")
            raise ExtractionError(f"Invalid ZIP file: {e}") from e

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise ExtractionError(f"Failed to extract and parse: {e}") from e

    def ingest_quarter(self, year: int, quarter: int) -> pd.DataFrame:
        """
        Download and process SEC data for a specific quarter.

        Convenience method that combines download_quarter and extract_and_parse.

        Args:
            year: The year (e.g., 2024).
            quarter: The quarter (1-4).

        Returns:
            DataFrame containing the parsed financial statement data.

        Example:
            >>> ingester = SECDataIngester()
            >>> df = ingester.ingest_quarter(2024, 3)
        """
        zip_path = self.download_quarter(year, quarter)
        return self.extract_and_parse(zip_path)

    def ingest_range(
        self,
        start_year: int,
        start_quarter: int,
        end_year: int,
        end_quarter: int,
    ) -> dict[str, pd.DataFrame]:
        """
        Download and process SEC data for a range of quarters.

        Args:
            start_year: Starting year.
            start_quarter: Starting quarter (1-4).
            end_year: Ending year.
            end_quarter: Ending quarter (1-4).

        Returns:
            Dictionary mapping quarter keys (e.g., "2024q3") to DataFrames.
        """
        results = {}

        # Generate list of quarters to process
        quarters_to_process = []
        current_year = start_year
        current_quarter = start_quarter

        while (current_year < end_year) or (
            current_year == end_year and current_quarter <= end_quarter
        ):
            quarters_to_process.append((current_year, current_quarter))
            current_quarter += 1
            if current_quarter > 4:
                current_quarter = 1
                current_year += 1

        logger.info(f"Processing {len(quarters_to_process)} quarters...")

        for year, quarter in tqdm(quarters_to_process, desc="Processing quarters"):
            key = f"{year}q{quarter}"
            try:
                results[key] = self.ingest_quarter(year, quarter)
                logger.info(f"Successfully processed {key}")
            except SECDataIngesterError as e:
                logger.error(f"Failed to process {key}: {e}")
                results[key] = None

        return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and process SEC EDGAR financial statement data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Q3 2024 data
  python ingest_sec_data.py --year 2024 --quarter 3

  # Download a range of quarters
  python ingest_sec_data.py --start-year 2023 --start-quarter 1 --end-year 2024 --end-quarter 2

  # Specify custom output directory
  python ingest_sec_data.py --year 2024 --quarter 3 --output-dir ./my_data
        """,
    )

    # Single quarter arguments
    parser.add_argument(
        "--year",
        "-y",
        type=int,
        help="Year for single quarter download (e.g., 2024)",
    )
    parser.add_argument(
        "--quarter",
        "-q",
        type=int,
        choices=[1, 2, 3, 4],
        help="Quarter for single download (1-4)",
    )

    # Range arguments
    parser.add_argument(
        "--start-year",
        type=int,
        help="Starting year for range download",
    )
    parser.add_argument(
        "--start-quarter",
        type=int,
        choices=[1, 2, 3, 4],
        help="Starting quarter for range download",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Ending year for range download",
    )
    parser.add_argument(
        "--end-quarter",
        type=int,
        choices=[1, 2, 3, 4],
        help="Ending quarter for range download",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for downloaded data (default: data/raw)",
    )

    # Other options
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download, skip extraction and parsing",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI."""
    args = parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate arguments
    single_quarter = args.year is not None and args.quarter is not None
    range_mode = all(
        [args.start_year, args.start_quarter, args.end_year, args.end_quarter]
    )

    if not single_quarter and not range_mode:
        logger.error(
            "Please specify either --year and --quarter for single download, "
            "or all range arguments (--start-year, --start-quarter, --end-year, --end-quarter)"
        )
        return 1

    if single_quarter and range_mode:
        logger.error("Cannot specify both single quarter and range arguments")
        return 1

    # Initialize ingester
    ingester = SECDataIngester(output_dir=args.output_dir, timeout=args.timeout)

    try:
        if single_quarter:
            logger.info(f"Processing Q{args.quarter} {args.year}...")

            if args.download_only:
                zip_path = ingester.download_quarter(args.year, args.quarter)
                logger.info(f"Downloaded: {zip_path}")
            else:
                df = ingester.ingest_quarter(args.year, args.quarter)
                logger.info(f"Successfully processed {len(df):,} records")

        else:
            logger.info(
                f"Processing range: Q{args.start_quarter} {args.start_year} "
                f"to Q{args.end_quarter} {args.end_year}..."
            )

            results = ingester.ingest_range(
                args.start_year,
                args.start_quarter,
                args.end_year,
                args.end_quarter,
            )

            # Summary
            successful = sum(1 for df in results.values() if df is not None)
            logger.info(f"Successfully processed {successful}/{len(results)} quarters")

        return 0

    except SECDataIngesterError as e:
        logger.error(f"Ingestion failed: {e}")
        return 1

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())

