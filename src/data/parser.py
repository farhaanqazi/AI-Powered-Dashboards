import os
import pandas as pd


def load_csv(file_storage):
    """
    file_storage: the uploaded file object from Flask (request.files["dataset"])
    returns: pandas DataFrame or None if something goes wrong
    """
    try:
        # Make sure we're at the start of the file
        if hasattr(file_storage, "seek"):
            file_storage.seek(0)
        elif hasattr(file_storage, "stream") and hasattr(file_storage.stream, "seek"):
            file_storage.stream.seek(0)

        # Try standard UTF-8 read first
        try:
            df = pd.read_csv(file_storage)
            return df
        except UnicodeDecodeError as e:
            print("Unicode error, retrying with latin1:", e)
            # Retry with a more forgiving encoding
            if hasattr(file_storage, "seek"):
                file_storage.seek(0)
            elif hasattr(file_storage, "stream") and hasattr(file_storage.stream, "seek"):
                file_storage.stream.seek(0)
            df = pd.read_csv(file_storage, encoding="latin1")
            return df

    except Exception as e:
        print("Error reading CSV:", e)
        return None


def load_csv_from_url(url: str):
    """
    Load a CSV directly from a URL (e.g. GitHub raw link).
    Returns a DataFrame or None on failure.
    """
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print("Error reading CSV from URL:", e)
        return None


def load_csv_from_kaggle(slug: str, csv_name: str | None = None):
    """
    Load a CSV from a Kaggle dataset using kagglehub.

    slug: e.g. "umitka/global-youth-unemployment-dataset"
    csv_name: optional specific CSV filename inside the dataset.
              If not provided, the first .csv file found will be used.

    Requires `pip install kagglehub` and Kaggle credentials configured
    in the environment.
    """
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Please 'pip install kagglehub' to use Kaggle sources.")
        return None

    try:
        path = kagglehub.dataset_download(slug)
        print("Downloaded Kaggle dataset to:", path)

        if csv_name:
            target = os.path.join(path, csv_name)
            if not os.path.isfile(target):
                print(f"CSV file '{csv_name}' not found in Kaggle dataset folder.")
                return None
            return pd.read_csv(target)

        # Otherwise, pick the first .csv file in the folder
        files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
        if not files:
            print("No CSV files found in Kaggle dataset folder.")
            return None

        first_csv = os.path.join(path, files[0])
        print("Using Kaggle CSV file:", first_csv)
        return pd.read_csv(first_csv)

    except Exception as e:
        print("Error loading Kaggle dataset:", e)
        return None
