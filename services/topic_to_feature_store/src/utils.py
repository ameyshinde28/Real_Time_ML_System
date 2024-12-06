from datetime import datetime, timezone
from src.logger import logger

def log_ohlc_to_elasticsearch(value: dict):
    
    logger.debug(f"Timestamp from the value dictionary: {value["timestamp_ms"]}")
    timestamp= datetime.fromtimestamp(value["timestamp_ms"] / 1000.0, tz=timezone.utc)

    logger.bind(
        timestamp=timestamp.isoformat(),
        product_id=value["product_id"], 
        price=value["close"]
        ).info(f'Received message: {value}')