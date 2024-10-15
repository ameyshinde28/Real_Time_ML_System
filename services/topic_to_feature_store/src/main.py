from typing import List, Optional

from quixstreams import Application
from loguru import logger

from src.hopsworks_api import push_value_to_feature_group

def topic_to_feature_store(
    kafka_broker_address: str,
    kafka_input_topic:str,
    kafka_consumer_group:str,
    feature_group_name:str,
    feature_group_version: int,
    feature_group_primary_keys: List[str],
    feature_group_event_time: str,
    start_offline_materialization: bool,
    batch_size: Optional[int] = None,
):
    """
    Reads incoming message from he given 'kafka_input_topic', and pushes them to the given 'feature_group_name' in the feature store.
    

    Args:
       kafka_broker_address (str): the address of the Kafka broker
        kafka_input_topic (str): the kafka topic to read the messages from
        kakfa_consumer_group (str): The kafka consumer group
        feature_group_name (str): The name of The Feature Group
        feature_gr oup_version (int): The vrsin of the Feature Group
        feature_group_primary_keys (List[str]): The primary keys of the Feature Group
        feature_group_event_time (str): The event time of the Feature Group
        start_offline_materialization (bool): Whether to start offline materialization
        batch_size (Optional[int]): The batch size to use for the offline materialization
        
    Returns:
        None
    """
    
    # Configure an Application. 
    # The config params will be used for the Consumer instance too.
    app = Application(
        broker_address=kafka_broker_address, 
        # auto_offset_reset='latest', 
        # auto_commit_enable=True,
        consumer_group=kafka_consumer_group
    )

    # Create a topic object for the input topic
    input_topic = app.topic(kafka_input_topic)

    batch = []

    # Create a consumer and start a polling loop
    with app.get_consumer() as consumer:
        consumer.subscribe(topics=[input_topic.name])

        while True:
            msg = consumer.poll(0.1)
            
            # breakpoint()
            
            if msg is None:
                continue
            elif msg.error():
                logger.error('Kafka error:', msg.error())
                continue

            value = msg.value()
            logger.debug(f'Received message: {value}')
            # Do some work with the value here ...
            import json
            value = json.loads(value.decode('utf-8'))
            
            logger.debug(f'Received message: {value}')
            
            # Append the value to the batch of trades
            batch.append(value)

            # If the length of the batch is not reached, continue
            if len(batch) < batch_size:
                logger.debug(f"Batch has size {len(batch)} < {batch_size}, Continue...")
                continue

            logger.debug(f"Batch has size {len(batch)} >= {batch_size}, Pushing data to feature store...")
            # If the batch size is reached, push the batch to the feature store
            push_value_to_feature_group(
                value=batch, 
                feature_group_name=feature_group_name, 
                feature_group_version=feature_group_version,
                feature_group_primary_keys=feature_group_primary_keys,
                feature_group_event_time=feature_group_event_time,
                start_offline_materialization=start_offline_materialization,
            )
            batch = []
            # breakpoint()
            
            # We need to push the data to the feature store
            
            
            
            # Store the offset of the processed message on the Consumer 
            # for the auto-commit mechanism.
            # It will send it to Kafka in the background.
            # Storing offset only after the message is processed enables at-least-once delivery
            # guarantees.
            consumer.store_offsets(message=msg)
            
            
if __name__ == '__main__':
    from config import config
    topic_to_feature_store(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        feature_group_name=config.feature_group_name,
        feature_group_version=config.feature_group_version,
        feature_group_primary_keys=config.feature_group_primary_keys,
        feature_group_event_time=config.feature_group_event_time,
        start_offline_materialization=config.start_offline_materialization,
        batch_size=config.batch_size,
    )