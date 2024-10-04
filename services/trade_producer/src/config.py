from pydantic_settings import BaseSettings
from typing import Optional

class AppConfig(BaseSettings):
    
    kafka_broker_address: str
    kafka_topic: str
    product_id: str
    live_or_historical: Optional[str] = None
    last_n_days: Optional[int] = None
    class Config:
        env_file = ".env"
        
config = AppConfig()
        