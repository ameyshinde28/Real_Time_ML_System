import pandas as pd


# Function to create a consistent hash of a pandas DataFrame
def hash_dataframe(df):
    return pd.util.hash_pandas_object(df).sum()

