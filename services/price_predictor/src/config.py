from pydantic_settings import BaseSettings

# class TrainingConfig(BaseSettings):
#     feature_view_name: str 
#     feature_view_version: int
#     ohlc_window_sec: int
#     forecast_steps: int
#     ml_model_status: str
#     feature_group_name: str
#     feature_group_version: int
#     product_id:str
#     last_n_days: int
#     n_search_trials: int
#     n_splits: int
#     last_n_minutes: int

#     class Config:
#         env_file = "training.env"

# class PredictionConfig(BaseSettings):
#     feature_view_name: str 
#     feature_view_version: int
#     ohlc_window_sec: int
#     forecast_steps: int
#     ml_model_status: str
#     api_supported_product_ids: list[str]

#     class Config:
#         env_file = "prediction.env"

class AppConfig(BaseSettings):
    feature_view_name: str 
    feature_view_version: int
    feature_group_name: str
    feature_group_version: int
    ohlc_window_sec: int
    product_id:str
    last_n_days: int
    forecast_steps: int
    n_search_trials: int
    n_splits: int
    last_n_minutes: int
    ml_model_status: str
    api_supported_product_ids: list[str]

    class Config:
        env_file = ".env"


class HopsworksConfig(BaseSettings):
    hopsworks_project_name: str
    hopsworks_api_key: str
    
    class Config:
        env_file = "hopsworks.credentials.env"

class CometConfig(BaseSettings):
    comet_ml_api_key: str
    comet_project_name: str
    comet_workspace: str

    class Config:
        env_file = "comet.credentials.env"

  
config = AppConfig()
# training_config = TrainingConfigConfig()
# prediction_config = PredictionConfig()
hopsworks_config = HopsworksConfig()
comet_config=CometConfig()