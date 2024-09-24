from src.kraken_websocket_api import KrakenWebSocketAPI, Trade
from loguru import logger

from typing import List

def produce_trades(
    kafka_broker_address: str,
    kafka_topic: str,
    product_id: str,
    
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
    
    from quixstreams import Application

    # Create an Application: 
    app = Application(broker_address=kafka_broker_address)
    
    
    # Define a Topic:
    topic = app.topic(name=kafka_topic, value_serializer='json')

    # Create a Kraken APi object
    kraken_api = KrakenWebSocketAPI(product_id=product_id)
    
     # Create a Producer and Produce Messages:

    with app.get_producer() as producer:
        
        while True:
                        
            trades: List[Trade] = kraken_api.get_trades()
            
            for trade in trades:
                # Serialize an event using the defined topic
                # Transform it into a sequence of bytes
                message = topic.serialize(key=trade.product_id, value=trade.model_dump())                
                # Produce a message into the Kafka topic
                producer.produce(topic=topic.name, value=message.value, key=message.key)
                logger.debug(f"Pushed to Kafka: {trade}")
                

if __name__ == "__main__":
    
    from src.config import config
    
    # print(f"product ids: {config.product_ids}")
    
    produce_trades(
        kafka_broker_address=config.kafka_broker_address, 
        kafka_topic=config.kafka_topic,
        product_id=config.product_id,
    )
    
    
    
    
    
    
    