from loguru import logger
from typing import List

from quixstreams import Application

from trade_data_source import TradeSource, Trade

def produce_trades(
    kafka_broker_address: str,
    kafka_topic: str,
    trade_data_source: TradeSource,
    
):
    """
    Reads trades from the Kraken Websocket API and saves them in the given
    'kafka topic'

    Args:
        kafka_broker_address (str): The address of the Kafka broker
        kafka_topic (str): The Kafka topic to save the trades
        product_id (str): The product id to get the trades from
        
    Returns:
        None
    """
    

    # Create an Application: 
    app = Application(broker_address=kafka_broker_address)
    
    
    # Define a Topic:
    topic = app.topic(name=kafka_topic, value_serializer='json')

    
     # Create a Producer and Produce Messages:

    with app.get_producer() as producer:
        
        while not trade_data_source.is_done():
                        
            trades: List[Trade] = trade_data_source.get_trades()
            
            for trade in trades:
                # Serialize an event using the defined topic
                # Transform it into a sequence of bytes
                message = topic.serialize(key=trade.product_id, value=trade.model_dump())                
                # Produce a message into the Kafka topic
                producer.produce(topic=topic.name, value=message.value, key=message.key)
                logger.debug(f"Pushed to Kafka: {trade}")
                

if __name__ == "__main__":
    
    from config import config
    
    from trade_data_source  import KrakenWebSocketAPI
    
    kraken_websocket_api = KrakenWebSocketAPI(
        product_id=config.product_id,
    )
    
    
    
    produce_trades(
        kafka_broker_address=config.kafka_broker_address, 
        kafka_topic=config.kafka_topic,
        trade_data_source=kraken_websocket_api,
    )
    
    
    
    
    
    
    