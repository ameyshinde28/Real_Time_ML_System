import pandas as pd
from datetime import datetime, timezone


# Function to create a consistent hash of a pandas DataFrame
def hash_dataframe(df):
    return pd.util.hash_pandas_object(df).sum()

def timestamp_ms_to_human_readable_utc(timestamp_ms: int) -> str:
        utc_datetime = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return utc_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

