import os
import pandas as pd
import logging
from typing import Optional, Union, NamedTuple
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tempfile

logger = logging.getLogger(__name__)

# Define maximum file size (e.g., 100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

class LoadResult(NamedTuple):
    """Structured result for data loading operations"""
    success: bool
    df: Optional[pd.DataFrame] = None
    error_code: Optional[str] = None
    detail: Optional[str] = None

def load_csv_from_file(file_storage, max_file_size: int = MAX_FILE_SIZE, max_rows: int = 50000) -> LoadResult:
    """
    Load a CSV file from Flask file storage with validation and error handling.

    Args:
        file_storage: the uploaded file object from Flask (request.files["dataset"])
        max_file_size: maximum allowed file size in bytes (enforcement recommended in main.py)
        max_rows: maximum number of rows to load (will sample if dataset is larger)

    Returns: LoadResult containing success status, DataFrame, and error details.
    """
    try:
        # Make sure we're at the start of the file
        if hasattr(file_storage, "seek"):
            file_storage.seek(0)
        elif hasattr(file_storage, "stream") and hasattr(file_storage.stream, "seek"):
            file_storage.stream.seek(0)

        # Check file size if possible (not always reliable from storage object)
        # Recommended to check in main.py using request size limits
        # if hasattr(file_storage, 'content_length') and file_storage.content_length > max_file_size:
        #     logger.error(f"File size {file_storage.content_length} exceeds limit {max_file_size}")
        #     return LoadResult(success=False, error_code="FILE_TOO_LARGE", detail="File size exceeds maximum allowed.")

        # Try standard UTF-8 read first with additional parameters for robustness
        try:
            # Try with automatic delimiter detection
            df = pd.read_csv(file_storage,
                           encoding='utf-8',
                           engine='python',
                           on_bad_lines='skip')  # Skip problematic lines instead of failing

            # Clean up column names - remove problematic characters and handle unnamed columns
            df.columns = [str(col).strip().replace(' ', '_').replace('-', '_') for col in df.columns]

            # Handle unnamed columns
            unnamed_count = 0
            new_columns = []
            for col in df.columns:
                if col.startswith('Unnamed:'):
                    new_columns.append(f'unnamed_col_{unnamed_count}')
                    unnamed_count += 1
                else:
                    new_columns.append(col)
            df.columns = new_columns

            logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")

            # Perform sampling if dataframe is too large
            if len(df) > max_rows:
                logger.info(f"Dataset has {len(df)} rows, sampling to {max_rows} rows for processing")
                df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
                logger.info(f"Sampled dataset now has {len(df)} rows")

            return LoadResult(success=True, df=df)
        except UnicodeDecodeError as e:
            logger.warning(f"Unicode error with UTF-8, retrying with latin1: {e}")
            # Retry with a more forgiving encoding
            if hasattr(file_storage, "seek"):
                file_storage.seek(0)
            elif hasattr(file_storage, "stream") and hasattr(file_storage.stream, "seek"):
                file_storage.stream.seek(0)
            df = pd.read_csv(file_storage,
                           encoding="latin1",
                           engine='python',
                           on_bad_lines='skip')

            # Clean up column names - remove problematic characters and handle unnamed columns
            df.columns = [str(col).strip().replace(' ', '_').replace('-', '_') for col in df.columns]

            # Handle unnamed columns
            unnamed_count = 0
            new_columns = []
            for col in df.columns:
                if col.startswith('Unnamed:'):
                    new_columns.append(f'unnamed_col_{unnamed_count}')
                    unnamed_count += 1
                else:
                    new_columns.append(col)
            df.columns = new_columns

            logger.info(f"Successfully loaded CSV with latin1 encoding")

            # Perform sampling if dataframe is too large
            if len(df) > max_rows:
                logger.info(f"Dataset has {len(df)} rows, sampling to {max_rows} rows for processing")
                df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
                logger.info(f"Sampled dataset now has {len(df)} rows")

            return LoadResult(success=True, df=df)
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty")
            return LoadResult(success=False, error_code="EMPTY_FILE", detail="The uploaded file is empty.")
        except pd.errors.ParserError as e:
            logger.error(f"Parser error while reading CSV: {e}")
            return LoadResult(success=False, error_code="PARSER_ERROR", detail=f"Could not parse the CSV file: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during CSV parsing: {e}")
            return LoadResult(success=False, error_code="UNEXPECTED_ERROR", detail=f"An unexpected error occurred: {str(e)}")

    except Exception as e:
        logger.exception(f"Unexpected error reading file object: {e}")
        return LoadResult(success=False, error_code="UNEXPECTED_FILE_ERROR", detail=f"An unexpected error occurred while reading the file: {str(e)}")


def load_csv_from_url(url: str, timeout: int = 30, max_rows: int = 50000) -> LoadResult:
    """
    Load a CSV directly from a URL (e.g. GitHub raw link).
    Includes validation, timeout, and error handling.

    Args:
        url: URL to the CSV file
        timeout: Request timeout in seconds

    Returns: LoadResult containing success status, DataFrame, and error details.
    """
    # Validate URL format
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            logger.error(f"Invalid URL format: {url}")
            return LoadResult(success=False, error_code="INVALID_URL", detail="The provided URL is invalid.")
        if parsed.scheme not in ['http', 'https']:
            logger.error(f"Invalid URL scheme: {parsed.scheme}")
            return LoadResult(success=False, error_code="INVALID_URL_SCHEME", detail="Only HTTP and HTTPS URLs are allowed.")
    except Exception as e:
        logger.error(f"Error parsing URL: {e}")
        return LoadResult(success=False, error_code="URL_PARSE_ERROR", detail="Could not parse the URL.")

    try:
        logger.info(f"Loading CSV from URL: {url}")

        # Configure session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        response = session.get(url, timeout=timeout)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx, 5xx)

        # Check if response is empty
        if not response.text.strip():
             logger.error(f"Response from URL is empty: {url}")
             return LoadResult(success=False, error_code="EMPTY_RESPONSE", detail="The URL returned an empty response.")

        # Read CSV from response content
        import io
        df = pd.read_csv(io.StringIO(response.text))
        logger.info(f"Successfully loaded CSV from URL with {len(df)} rows and {len(df.columns)} columns")

        # Perform sampling if dataframe is too large
        if len(df) > max_rows:
            logger.info(f"Dataset has {len(df)} rows, sampling to {max_rows} rows for processing")
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled dataset now has {len(df)} rows")

        return LoadResult(success=True, df=df)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV from URL is empty: {url}")
        return LoadResult(success=False, error_code="EMPTY_CSV_URL", detail="The CSV file at the URL is empty.")
    except pd.errors.ParserError as e:
        logger.error(f"Parser error loading CSV from URL {url}: {e}")
        return LoadResult(success=False, error_code="URL_PARSER_ERROR", detail=f"Could not parse the CSV file from the URL: {str(e)}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error loading CSV from URL {url}: {e}")
        status_code = e.response.status_code if e.response else "Unknown"
        return LoadResult(success=False, error_code="HTTP_ERROR", detail=f"HTTP {status_code} error occurred while fetching the URL: {str(e)}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error loading CSV from URL {url}: {e}")
        return LoadResult(success=False, error_code="CONNECTION_ERROR", detail="Could not connect to the URL. Please check the address and your network connection.")
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout error loading CSV from URL {url}: {e}")
        return LoadResult(success=False, error_code="TIMEOUT_ERROR", detail=f"The request timed out after {timeout} seconds.")
    except requests.exceptions.RequestException as e: # Catch other requests-related errors
        logger.error(f"Request error loading CSV from URL {url}: {e}")
        return LoadResult(success=False, error_code="REQUEST_ERROR", detail=f"A network request error occurred: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error loading CSV from URL {url}: {e}")
        return LoadResult(success=False, error_code="UNEXPECTED_URL_ERROR", detail=f"An unexpected error occurred: {str(e)}")


def load_csv_from_kaggle(slug: str, csv_name: Optional[str] = None, timeout: int = 60, max_rows: int = 50000) -> LoadResult:
    """
    Load a CSV from a Kaggle dataset using kagglehub with validation and error handling.

    Args:
        slug: e.g. "umitka/global-youth-unemployment-dataset"
        csv_name: optional specific CSV filename inside the dataset.
                  If not provided, the first .csv file found will be used.
        timeout: timeout for download operation in seconds (Note: kagglehub may not respect this directly)

    Requires `pip install kagglehub` and Kaggle credentials configured
    in the environment.
    """
    # Validate slug format (basic validation)
    if not slug or '/' not in slug:
        logger.error(f"Invalid Kaggle dataset slug format: {slug}")
        return LoadResult(success=False, error_code="INVALID_SLUG", detail="The Kaggle dataset slug format is invalid (expected 'username/dataset').")

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub is not installed. Please 'pip install kagglehub' to use Kaggle sources.")
        return LoadResult(success=False, error_code="KAGGLEHUB_MISSING", detail="kagglehub library is not installed. Please install it using 'pip install kagglehub'.")

    temp_dir = None
    try:
        logger.info(f"Downloading Kaggle dataset: {slug}")
        # Use kagglehub.dataset_download which returns a path to the local dataset directory
        path = kagglehub.dataset_download(slug, verbose=False)
        logger.info(f"Downloaded Kaggle dataset to: {path}")

        if csv_name:
            target = os.path.join(path, csv_name)
            if not os.path.isfile(target):
                logger.error(f"CSV file '{csv_name}' not found in Kaggle dataset folder: {path}")
                return LoadResult(success=False, error_code="CSV_NOT_FOUND", detail=f"Specific CSV file '{csv_name}' not found in the dataset.")
            df = pd.read_csv(target)
            logger.info(f"Successfully loaded specific CSV file from Kaggle dataset: {csv_name}")

            # Perform sampling if dataframe is too large
            if len(df) > max_rows:
                logger.info(f"Dataset has {len(df)} rows, sampling to {max_rows} rows for processing")
                df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
                logger.info(f"Sampled dataset now has {len(df)} rows")

            return LoadResult(success=True, df=df)

        # Otherwise, pick the first .csv file in the folder
        files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
        if not files:
            logger.error("No CSV files found in Kaggle dataset folder.")
            return LoadResult(success=False, error_code="NO_CSV_FILES", detail="No CSV files were found in the downloaded dataset.")

        first_csv = os.path.join(path, files[0])
        logger.info(f"Loading first CSV file from Kaggle dataset: {files[0]}")
        df = pd.read_csv(first_csv)
        logger.info(f"Successfully loaded CSV from Kaggle dataset with {len(df)} rows and {len(df.columns)} columns")

        # Perform sampling if dataframe is too large
        if len(df) > max_rows:
            logger.info(f"Dataset has {len(df)} rows, sampling to {max_rows} rows for processing")
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled dataset now has {len(df)} rows")

        return LoadResult(success=True, df=df)

    except requests.exceptions.HTTPError as e:
        # kagglehub might raise HTTP errors for auth/dataset not found
        status_code = e.response.status_code if e.response else "Unknown"
        logger.error(f"HTTP error downloading Kaggle dataset {slug}: {e}")
        if status_code == 404:
            return LoadResult(success=False, error_code="DATASET_NOT_FOUND", detail="The specified Kaggle dataset was not found.")
        elif status_code == 401:
            return LoadResult(success=False, error_code="KAGGLE_AUTH_ERROR", detail="Authentication failed. Please check your Kaggle credentials.")
        else:
            return LoadResult(success=False, error_code="KAGGLE_HTTP_ERROR", detail=f"An HTTP error occurred while downloading the dataset (Status {status_code}): {str(e)}")

    except Exception as e:
        logger.exception(f"Error loading Kaggle dataset: {e}")
        # Provide more specific error messages based on common issues
        if "not found" in str(e).lower():
            return LoadResult(success=False, error_code="DATASET_NOT_FOUND", detail="The specified Kaggle dataset was not found.")
        elif "permission" in str(e).lower():
             return LoadResult(success=False, error_code="KAGGLE_PERMISSION_ERROR", detail="Permission denied. Please check your Kaggle credentials and dataset access.")
        else:
            return LoadResult(success=False, error_code="KAGGLE_DOWNLOAD_ERROR", detail=f"An error occurred while downloading or processing the Kaggle dataset: {str(e)}")

    finally:
        # Optional: Clean up temporary directory if kagglehub created one internally
        # This is tricky as kagglehub manages its own cache. We rely on its internal cleanup.
        # If we were creating our own temp dir explicitly, we would remove it here.
        pass
