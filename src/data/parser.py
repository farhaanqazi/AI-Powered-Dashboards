import os
import pandas as pd
import logging
from typing import Optional, Union, NamedTuple
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Define maximum file size (e.g., 100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

class LoadResult(NamedTuple):
    """Structured result for data loading operations"""
    success: bool
    df: Optional[pd.DataFrame] = None
    error_code: Optional[str] = None
    detail: Optional[str] = None

def load_csv(file_storage, max_file_size: int = MAX_FILE_SIZE) -> Optional[pd.DataFrame]:
    """
    Load a CSV file from Flask file storage with validation and error handling.

    Args:
        file_storage: the uploaded file object from Flask (request.files["dataset"])
        max_file_size: maximum allowed file size in bytes (not enforced to avoid upload issues)

    Returns: pandas DataFrame or None if something goes wrong
    """
    try:
        # Make sure we're at the start of the file
        if hasattr(file_storage, "seek"):
            file_storage.seek(0)
        elif hasattr(file_storage, "stream") and hasattr(file_storage.stream, "seek"):
            file_storage.stream.seek(0)

        # Try standard UTF-8 read first with additional parameters for robustness
        try:
            # Try with automatic delimiter detection
            df = pd.read_csv(file_storage,
                           encoding='utf-8',
                           engine='python',
                           on_bad_lines='skip')  # Skip problematic lines instead of failing
            logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
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
            logger.info(f"Successfully loaded CSV with latin1 encoding")
            return df
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty")
            return None
        except pd.errors.ParserError as e:
            logger.error(f"Parser error while reading CSV: {e}")
            return None

    except Exception as e:
        logger.exception(f"Unexpected error reading CSV: {e}")
        return None


def load_csv_from_url(url: str, timeout: int = 30) -> Optional[pd.DataFrame]:
    """
    Load a CSV directly from a URL (e.g. GitHub raw link).
    Includes validation, timeout, and error handling.

    Args:
        url: URL to the CSV file
        timeout: Request timeout in seconds

    Returns: pandas DataFrame or None on failure.
    """
    # Validate URL format
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            logger.error(f"Invalid URL format: {url}")
            return None
        if parsed.scheme not in ['http', 'https']:
            logger.error(f"Invalid URL scheme: {parsed.scheme}")
            return None
    except Exception as e:
        logger.error(f"Error parsing URL: {e}")
        return None

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
        response.raise_for_status()  # Raise an exception for bad status codes

        # Read CSV from response content
        import io
        df = pd.read_csv(io.StringIO(response.text))
        logger.info(f"Successfully loaded CSV from URL with {len(df)} rows and {len(df.columns)} columns")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"CSV from URL is empty: {url}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Parser error loading CSV from URL {url}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error loading CSV from URL {url}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Error loading CSV from URL {url}: {e}")
        return None


def load_csv_from_kaggle(slug: str, csv_name: Optional[str] = None, timeout: int = 60) -> Optional[pd.DataFrame]:
    """
    Load a CSV from a Kaggle dataset using kagglehub with validation and error handling.

    Args:
        slug: e.g. "umitka/global-youth-unemployment-dataset"
        csv_name: optional specific CSV filename inside the dataset.
                  If not provided, the first .csv file found will be used.
        timeout: timeout for download operation in seconds

    Requires `pip install kagglehub` and Kaggle credentials configured
    in the environment.
    """
    # Validate slug format (basic validation)
    if not slug or '/' not in slug:
        logger.error(f"Invalid Kaggle dataset slug format: {slug}")
        return None

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub is not installed. Please 'pip install kagglehub' to use Kaggle sources.")
        return None

    try:
        logger.info(f"Downloading Kaggle dataset: {slug}")
        path = kagglehub.dataset_download(slug, verbose=False)
        logger.info(f"Downloaded Kaggle dataset to: {path}")

        if csv_name:
            target = os.path.join(path, csv_name)
            if not os.path.isfile(target):
                logger.error(f"CSV file '{csv_name}' not found in Kaggle dataset folder: {path}")
                return None
            df = pd.read_csv(target)
            logger.info(f"Successfully loaded specific CSV file from Kaggle dataset: {csv_name}")
            return df

        # Otherwise, pick the first .csv file in the folder
        files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
        if not files:
            logger.error("No CSV files found in Kaggle dataset folder.")
            return None

        first_csv = os.path.join(path, files[0])
        logger.info(f"Loading first CSV file from Kaggle dataset: {files[0]}")
        df = pd.read_csv(first_csv)
        logger.info(f"Successfully loaded CSV from Kaggle dataset with {len(df)} rows and {len(df.columns)} columns")
        return df

    except Exception as e:
        logger.exception(f"Error loading Kaggle dataset: {e}")
        return None
