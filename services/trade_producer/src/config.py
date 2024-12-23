from pydantic_settings import BaseSettings
from typing import Optional

class AppConfig(BaseSettings):
    
    kafka_broker_address: Optional[str] = None
    kafka_topic: str
    product_ids: list[str]
    live_or_historical: Optional[str] = None
    last_n_days: Optional[int] = None
    class Config:
        env_file = ".env"
        
config = AppConfig()
        