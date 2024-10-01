from pydantic_settings import BaseSettings

class AppConfig(BaseSettings):
    kafka_broker_address:str
    kafka_input_topic:str
    kafka_output_topic:str
    kafka_consumer_group:str
    ohlcv_window_seconds:int
    
    class Config:
        env_file="ohlcv.env"

config=AppConfig()