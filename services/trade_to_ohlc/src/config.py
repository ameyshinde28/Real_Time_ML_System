from pydantic_settings import BaseSettings
from typing import Optional
class AppConfig(BaseSettings):
    kafka_broker_address:Optional[str]=None
    kafka_input_topic:str
    kafka_output_topic:str
    kafka_consumer_group:str
    ohlcv_window_seconds:int
    
    class Config:
        env_file=".env"

config=AppConfig()