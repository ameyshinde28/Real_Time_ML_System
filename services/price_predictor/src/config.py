from pydantic_settings import BaseSettings

class AppConfig(BaseSettings):
    feature_view_name: str 
    feature_view_version: int
    feature_group_name: str
    feature_group_version: int
    ohlc_window_sec: int
    product_id:str
    last_n_days: int
    forecast_steps: int

    class Config:
        env_file = ".env"


class HopsworksConfig(BaseSettings):
    hopsworks_project_name: str
    hopsworks_api_key: str
    
    class Config:
        env_file = "credentials.env"
        
        
config = AppConfig()
hopsworks_config = HopsworksConfig()
