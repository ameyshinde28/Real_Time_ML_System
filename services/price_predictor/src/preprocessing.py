import pandas as pd
def keep_only_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    return df[['open', 'high', 'low', 'close', 'volume', 'timestamp_ms']]