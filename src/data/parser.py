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
