from quixstreams import Application
from loguru import logger
from datetime import timedelta



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

    return candle


def transform_trade_to_ohlcv(
    kafka_broker_address: str,
    kafka_input_topic:str,
    kafka_output_topic:str,
    kafka_consumer_group: str,
    ohlcv_window_seconds = 60,
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
    
    input_topic = app.topic(name=kafka_input_topic, value_deserializer='json')
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
    sdf['timestamp_ms'] = sdf['value']['timestamp_ms']
    
    
    # keep the columns we are intrested
    sdf = sdf[['timestamp_ms', 'close', 'low', 'high', 'close', 'volume']]
    
    
    # print the output to the console
    sdf.update(logger.debug)
    
    # push these message to the output topic
    sdf = sdf.to_topic(output_topic)    
    app.run(sdf)
    
    
if __name__ == '__main__':
    
    transform_trade_to_ohlcv(
        kafka_broker_address='localhost:19092',
        kafka_input_topic = 'trades',
        kafka_output_topic='ohlcv',
        kafka_consumer_group='consumer_group_trade_to_ohlcv',
        ohlcv_window_seconds = 60,
    )
            