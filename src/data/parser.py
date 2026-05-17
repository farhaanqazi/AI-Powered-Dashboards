import ipaddress
import os
import socket
import pandas as pd
import logging
from typing import Optional, Union, NamedTuple, List, Tuple
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tempfile

from src import config

logger = logging.getLogger(__name__)

# Define maximum file size (e.g., 100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024


class SSRFError(Exception):
    """Raised when a URL fails SSRF validation."""


def _ip_is_blocked(ip: str) -> bool:
    """Block loopback, private, link-local (incl. 169.254.169.254 metadata),
    reserved, multicast and unspecified addresses."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _resolve_and_check_host(host: str) -> None:
    """Resolve every A/AAAA record for ``host`` and reject if any address is
    non-public. Defends against DNS-rebinding-style payloads at request time."""
    if config.ALLOW_PRIVATE_URL_FETCH:
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise SSRFError(f"Could not resolve host '{host}': {exc}")
    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise SSRFError(f"Host '{host}' resolved to no addresses.")
    for ip in addresses:
        if _ip_is_blocked(ip):
            raise SSRFError(
                f"Refusing to fetch '{host}': resolves to non-public address {ip}."
            )


def _validate_fetch_url(url: str) -> None:
    """Scheme allow-list + private/metadata IP block. Raises SSRFError."""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in config.URL_FETCH_ALLOWED_SCHEMES:
        raise SSRFError(f"Disallowed URL scheme '{parsed.scheme}'.")
    host = parsed.hostname
    if not host:
        raise SSRFError("URL has no host.")
    # If the host is a literal IP, check it directly; else resolve + check.
    try:
        ipaddress.ip_address(host)
        if not config.ALLOW_PRIVATE_URL_FETCH and _ip_is_blocked(host):
            raise SSRFError(f"Refusing to fetch non-public address {host}.")
    except ValueError:
        _resolve_and_check_host(host)


def _ssrf_safe_get(url: str):
    """GET ``url`` with redirects followed manually, re-validating every hop,
    capping redirect count, enforcing a hard byte cap while streaming."""
    timeout = config.URL_FETCH_TIMEOUT_SECONDS
    max_bytes = config.URL_FETCH_MAX_BYTES
    current = url
    session = requests.Session()
    retry_strategy = Retry(total=0)
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    for _ in range(config.URL_FETCH_MAX_REDIRECTS + 1):
        _validate_fetch_url(current)
        resp = session.get(
            current, timeout=timeout, allow_redirects=False, stream=True
        )
        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            resp.close()
            if not loc:
                raise SSRFError("Redirect without Location header.")
            current = requests.compat.urljoin(current, loc)
            continue
        resp.raise_for_status()
        body = bytearray()
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            body.extend(chunk)
            if len(body) > max_bytes:
                resp.close()
                raise SSRFError(
                    f"Remote file exceeds {max_bytes} byte cap."
                )
        ctype = (resp.headers.get("Content-Type") or "").lower()
        resp.close()
        return bytes(body), ctype
    raise SSRFError(
        f"Exceeded {config.URL_FETCH_MAX_REDIRECTS} redirect hops."
    )

class LoadResult(NamedTuple):
    """Structured result for data loading operations"""
    success: bool
    df: Optional[pd.DataFrame] = None
    error_code: Optional[str] = None
    detail: Optional[str] = None
    warnings: Optional[List[str]] = None

def _try_detect_encoding(sample: bytes) -> Tuple[Optional[str], Optional[float]]:
    try:
        import chardet
    except Exception:
        return None, None

    try:
        result = chardet.detect(sample)
        return result.get("encoding"), result.get("confidence")
    except Exception:
        return None, None

def _read_sample_bytes(file_storage, size: int = 65536) -> bytes:
    try:
        if hasattr(file_storage, "seek"):
            file_storage.seek(0)
        data = file_storage.read(size)
        if hasattr(file_storage, "seek"):
            file_storage.seek(0)
        return data or b""
    except Exception:
        return b""

def load_csv_from_file(
    file_storage,
    max_file_size: int = MAX_FILE_SIZE,
    max_rows: int = 500000,
    encoding: Optional[str] = None,
) -> LoadResult:
    """
    Load a CSV file from Flask file storage with validation and error handling.

    Args:
        file_storage: the uploaded file object from Flask (request.files["dataset"])
        max_file_size: maximum allowed file size in bytes (enforcement recommended in main.py)
        max_rows: maximum number of rows to load (will sample if dataset is larger)

    Returns: LoadResult containing success status, DataFrame, and error details.
    """
    warnings: List[str] = []
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

        # Determine encoding (user override takes precedence)
        detected_encoding = None
        confidence = None
        if encoding:
            detected_encoding = encoding
        else:
            sample = _read_sample_bytes(file_storage)
            detected_encoding, confidence = _try_detect_encoding(sample)
            if detected_encoding is None:
                warnings.append("Encoding detection unavailable; defaulting to UTF-8.")
                detected_encoding = "utf-8"
            elif confidence is not None and confidence < 0.6:
                warnings.append(f"Low encoding confidence ({confidence:.2f}) for '{detected_encoding}'.")

        # Try reading with detected/override encoding first
        try:
            df = pd.read_csv(
                file_storage,
                encoding=detected_encoding,
                engine='python',
                on_bad_lines='skip'
            )

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

            return LoadResult(success=True, df=df, warnings=warnings)
        except UnicodeDecodeError as e:
            if encoding:
                logger.error(f"Unicode error with provided encoding '{encoding}': {e}")
                return LoadResult(success=False, error_code="ENCODING_ERROR", detail=f"Failed to decode with encoding '{encoding}': {e}")

            warnings.append(f"Unicode error with '{detected_encoding}', retrying with latin1.")
            logger.warning(f"Unicode error with {detected_encoding}, retrying with latin1: {e}")
            if hasattr(file_storage, "seek"):
                file_storage.seek(0)
            elif hasattr(file_storage, "stream") and hasattr(file_storage.stream, "seek"):
                file_storage.stream.seek(0)
            df = pd.read_csv(
                file_storage,
                encoding="latin1",
                engine='python',
                on_bad_lines='skip'
            )

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

            return LoadResult(success=True, df=df, warnings=warnings)
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty")
            return LoadResult(success=False, error_code="EMPTY_FILE", detail="The uploaded file is empty.", warnings=warnings)
        except pd.errors.ParserError as e:
            logger.error(f"Parser error while reading CSV: {e}")
            return LoadResult(success=False, error_code="PARSER_ERROR", detail=f"Could not parse the CSV file: {str(e)}", warnings=warnings)
        except Exception as e:
            logger.error(f"Unexpected error during CSV parsing: {e}")
            return LoadResult(success=False, error_code="UNEXPECTED_ERROR", detail=f"An unexpected error occurred: {str(e)}", warnings=warnings)

    except Exception as e:
        logger.exception(f"Unexpected error reading file object: {e}")
        return LoadResult(success=False, error_code="UNEXPECTED_FILE_ERROR", detail=f"An unexpected error occurred while reading the file: {str(e)}", warnings=warnings)


def load_csv_from_url(url: str, timeout: int = 30, max_rows: int = 500000) -> LoadResult:
    """
    Load a CSV directly from a URL (e.g. GitHub raw link).
    Includes validation, timeout, and error handling.

    Args:
        url: URL to the CSV file
        timeout: Request timeout in seconds

    Returns: LoadResult containing success status, DataFrame, and error details.
    """
    # Validate URL format + SSRF posture (scheme allow-list, private/metadata
    # IP block, redirect cap, byte cap, content-type sniff before parse).
    try:
        _validate_fetch_url(url)
    except SSRFError as e:
        logger.error(f"SSRF validation rejected URL {url}: {e}")
        return LoadResult(success=False, error_code="SSRF_BLOCKED", detail=str(e))
    except Exception as e:
        logger.error(f"Error parsing URL: {e}")
        return LoadResult(success=False, error_code="URL_PARSE_ERROR", detail="Could not parse the URL.")

    warnings: List[str] = []
    try:
        logger.info(f"Loading CSV from URL: {url}")

        content, ctype = _ssrf_safe_get(url)

        # Check if response is empty
        if not content.strip():
            logger.error(f"Response from URL is empty: {url}")
            return LoadResult(success=False, error_code="EMPTY_RESPONSE", detail="The URL returned an empty response.")

        # Content-type sniff BEFORE parse: reject HTML/markup masquerading as CSV.
        if any(t in ctype for t in ("text/html", "application/xhtml", "application/xml", "text/xml")):
            logger.error(f"URL returned non-CSV content-type '{ctype}': {url}")
            return LoadResult(success=False, error_code="NOT_CSV_CONTENT", detail=f"URL returned '{ctype.split(';')[0]}', not CSV.")
        head = content[:4096].lstrip()
        if head[:9].lower() in (b"<!doctype", b"<html><he") or head[:5].lower() == b"<html" or head[:5] == b"<?xml":
            logger.error(f"URL returned HTML/XML body, not CSV: {url}")
            return LoadResult(success=False, error_code="NOT_CSV_CONTENT", detail="URL returned HTML/XML, not CSV.")

        # Detect encoding from content
        detected_encoding, confidence = _try_detect_encoding(content)
        if detected_encoding is None:
            detected_encoding = "utf-8"
            warnings.append("Encoding detection unavailable; defaulting to UTF-8.")
        elif confidence is not None and confidence < 0.6:
            warnings.append(f"Low encoding confidence ({confidence:.2f}) for '{detected_encoding}'.")

        # Read CSV from response content
        import io
        text = content.decode(detected_encoding, errors="replace")
        df = pd.read_csv(io.StringIO(text))
        logger.info(f"Successfully loaded CSV from URL with {len(df)} rows and {len(df.columns)} columns")

        # Perform sampling if dataframe is too large
        if len(df) > max_rows:
            logger.info(f"Dataset has {len(df)} rows, sampling to {max_rows} rows for processing")
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled dataset now has {len(df)} rows")

        return LoadResult(success=True, df=df, warnings=warnings)
    except pd.errors.EmptyDataError:
        logger.error(f"CSV from URL is empty: {url}")
        return LoadResult(success=False, error_code="EMPTY_CSV_URL", detail="The CSV file at the URL is empty.", warnings=warnings)
    except pd.errors.ParserError as e:
        logger.error(f"Parser error loading CSV from URL {url}: {e}")
        return LoadResult(success=False, error_code="URL_PARSER_ERROR", detail=f"Could not parse the CSV file from the URL: {str(e)}", warnings=warnings)
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error loading CSV from URL {url}: {e}")
        status_code = e.response.status_code if e.response else "Unknown"
        return LoadResult(success=False, error_code="HTTP_ERROR", detail=f"HTTP {status_code} error occurred while fetching the URL: {str(e)}", warnings=warnings)
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error loading CSV from URL {url}: {e}")
        return LoadResult(success=False, error_code="CONNECTION_ERROR", detail="Could not connect to the URL. Please check the address and your network connection.", warnings=warnings)
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout error loading CSV from URL {url}: {e}")
        return LoadResult(success=False, error_code="TIMEOUT_ERROR", detail=f"The request timed out after {timeout} seconds.", warnings=warnings)
    except requests.exceptions.RequestException as e: # Catch other requests-related errors
        logger.error(f"Request error loading CSV from URL {url}: {e}")
        return LoadResult(success=False, error_code="REQUEST_ERROR", detail=f"A network request error occurred: {str(e)}", warnings=warnings)
    except SSRFError as e:
        logger.error(f"SSRF block during fetch of {url}: {e}")
        return LoadResult(success=False, error_code="SSRF_BLOCKED", detail=str(e), warnings=warnings)
    except Exception as e:
        logger.exception(f"Unexpected error loading CSV from URL {url}: {e}")
        return LoadResult(success=False, error_code="UNEXPECTED_URL_ERROR", detail=f"An unexpected error occurred: {str(e)}", warnings=warnings)


def load_csv_from_kaggle(slug: str, csv_name: Optional[str] = None, timeout: int = 60, max_rows: int = 500000) -> LoadResult:
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
