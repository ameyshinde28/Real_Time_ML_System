from quixstreams import Application
from loguru import logger
from datetime import timedelta
from typing import Any, Optional, List, Tuple
# from quixstreams.types import TimestampType


def init_ohlcv_candle(trade: dict):
    """Returns the initial OLHC candle when the first 'trade' in that window is happens.

    Args:
        trade (dict): A dictionary of trades
    """
    return{
        'open' : trade['price'],
        'high' : trade['price'], 
        'low' : trade['price'],
        'close' : trade['price'],
        'volume' : trade['price'],
        'product_id' : trade['product_id']
    }

def update_ohlcv_candle(candle:dict, trade: dict):
    """Updates the OHLCV candle with the new 'trade'.

    Args:
        candle (dict): _description_
        reade (dict): _description_
        
    """
    candle['high'] = max(candle['high'], trade['price'])
    candle['low'] = min(candle['low'], trade['price'])
    candle['close'] = trade['price']
    candle['volume'] += trade['quantity'] 
    candle['product_id'] = trade['product_id']
       

    return candle

def custom_ts_extractor(
    value: Any,
    headers: Optional[List[Tuple[str, bytes]]],
    timestamp: float,
    timestamp_type: Optional[str] = None,
) -> int:
    """
    Specifying a custom timestamp extractor to use the timestamp from the message payload 
    instead of Kafka timestamp.
    Extracts the timestamp from the message payload.
    """
    return value["timestamp_ms"]


def transform_trade_to_ohlcv(
    kafka_broker_address: str,
    kafka_input_topic:str,
    kafka_output_topic:str,
    kafka_consumer_group: str,
    ohlcv_window_seconds:int,
):
    """reads incoming trades from the 'kafka_input_topic'
    , transforms them into OHLC data and outputs them to the 
    given 'kafka_output_topic'.
    Args:
        kafka_broker_address (str): The address of the Kafka broker
        kafka_input_topic (str): The Kafka topic to read the trades fro,
        kafka_output_topic (str): The Kafka topic to save the OHLC data
        kafka_consumer_group (str): The Kafka consumer group
    
    Returns:
        None
    """
    
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
    )
    
    
    
    input_topic = app.topic(name=kafka_input_topic, value_deserializer='json', timestamp_extractor=custom_ts_extractor)
    output_topic = app.topic(name=kafka_output_topic, value_deserializer='json')
    
    # Create a Quix Streams dataframe
    sdf = app.dataframe(input_topic)
    
    # sdf.update(logger.debug)
    
    
    sdf = (sdf.tumbling_window(duration_ms=timedelta(seconds=ohlcv_window_seconds))
    .reduce(reducer=update_ohlcv_candle, initializer=init_ohlcv_candle)
    .final()
    # .current()
    )
    
    # Unpack the dictionary into separate columns
    sdf['open'] = sdf['value']['open']
    sdf['high'] = sdf['value']['high']
    sdf['low'] = sdf['value']['low']
    sdf['close'] = sdf['value']['close']
    sdf['volume'] = sdf['value']['volume']
    sdf['product_id'] = sdf['value']['product_id']
    sdf['timestamp_ms'] = sdf['end']
    
    # keep the columns we are intrested
    sdf = sdf[['product_id','timestamp_ms', 'open', 'low', 'high', 'close', 'volume']]
    
    
    # print the output to the console
    sdf.update(logger.debug)
    
    # push these message to the output topic
    sdf = sdf.to_topic(output_topic)    
    app.run(sdf)
    
    
if __name__ == '__main__':
    
    from config import config
    
    # transform_trade_to_ohlcv(
    #     kafka_broker_address = "localhost:19092",
    #     kafka_input_topic = "trades",
    #     kafka_output_topic = "olhcv",
    #     kafka_consumer_group = "trades_to_olhcv",
    #     ohlcv_window_seconds = 60,
    # )
    
    transform_trade_to_ohlcv(
        kafka_broker_address = config.kafka_broker_address,
        kafka_input_topic = config.kafka_input_topic,
        kafka_output_topic = config.kafka_output_topic,
        kafka_consumer_group = config.kafka_consumer_group,
        ohlcv_window_seconds = config.ohlcv_window_seconds,
    )
            