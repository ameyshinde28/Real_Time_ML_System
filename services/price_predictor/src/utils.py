import pandas as pd
from datetime import datetime, timezone
import os
# from price_predictor import PricePrediction
from src.logger import logger

# Function to create a consistent hash of a pandas DataFrame
def hash_dataframe(df):
    return pd.util.hash_pandas_object(df).sum()

def timestamp_ms_to_human_readable_utc(timestamp_ms: int) -> str:
        utc_datetime = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return utc_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

def get_git_commit_hash() -> str:
    """
    Get the git commit hash.

    Returns:
        str: The git commit hash.
    """
    import subprocess

    try:
        git_commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        git_commit_hash = os.environ.get('GIT_COMMIT_HASH', 'Unknown')
    
    return git_commit_hash

def log_prediction_to_elasticsearch(prediction: 'PricePrediction'):
    """_summary_

    Args:
        prediction (PricePrediction): _description_
    """
    timestamp= datetime \
    .fromtimestamp(prediction.timestamp_ms / 1000.0, tz=timezone.utc) \
    .isoformat()

    logger.bind(
        timestamp=timestamp,
        product_id=prediction.product_id, 
        price=prediction.price
    ).info(f'Prediction: {prediction.to_json()}',)