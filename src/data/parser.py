import pandas as pd


def load_csv(file_storage):
    """
    file_storage: the uploaded file object from Flask (request.files["dataset"])
    returns: pandas DataFrame or None if something goes wrong
    """
    try:
        # file_storage behaves like a file object, so we can pass it directly
        df = pd.read_csv(file_storage)
        return df
    except Exception as e:
        print("Error reading CSV:", e)
        return None
