from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    
    kafka_broker_address:str
    kafka_topic:str
    product_id:str
    
    class Config:
        env_file = ".env"
        
config = AppConfig()
        