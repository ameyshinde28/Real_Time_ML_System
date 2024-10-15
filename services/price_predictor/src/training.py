from src.config import HopsworksConfig
from typing import Optional
from sklearn.metrics import mean_absolute_error

from loguru import logger
def train_model(
    feature_view_name: str,
    feature_view_version: int,
    feature_group_name: str,
    feature_group_version: int,
    ohlc_window_sec: int,
    product_id: str,
    last_n_days:int,
    hopsworks_config: HopsworksConfig,
    forecast_steps: int,
    prec_test_data: Optional[float] = 0.3
    
):
    """
    Read data from the feature store
    Trains a predictive model
    Saves the model to the model registry
    Args:
    Returns:
    """
    # Load the data from the feature store
    from src.ohlc_data_reader import OhlcDataReader
    
    ohlcv_data_reader = OhlcDataReader(
        hopsworks_config=hopsworks_config,
        feature_view_name=feature_view_name,
        feature_view_version=feature_view_version,
        feature_group_name=feature_group_name,
        feature_group_version=feature_group_version,
        ohlc_window_sec=ohlc_window_sec,
        
    )
    # Read tthe sorted data from the offline store
    # data is sorted by timestamp_ms
    ohlc_data=ohlcv_data_reader.read_from_offline_store(
        product_id=product_id,
        last_n_days=last_n_days)
    
    logger.debug(f"Read {len(ohlc_data)} rows of data from the offline store")
    
    # split the data into training and testing
    logger.debug(f"Splitting the data into training and testing")
    test_size = int(len(ohlc_data) * prec_test_data)
    train_df = ohlc_data[:-test_size]
    test_df = ohlc_data[-test_size:]
    
    # Add a column with the target  price we want our model to predict
    train_df['target_price'] = train_df['close'].shift(-forecast_steps)
    test_df['target_price'] = test_df['close'].shift(-forecast_steps)
    logger.debug(f"Added target price column to training and testing data")
    
    # remove row with NaN values
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    logger.debug(f"Removed rows with NaN values")
    logger.debug(f"Training data after removing NaN values: {len(train_df)}")
    logger.debug(f"Testing data after removing NaN values: {len(test_df)}")
    
    
    # Split the data into features and target
    X_train = train_df.drop(columns='target_price')
    y_train = train_df['target_price']
    X_test = test_df.drop(columns='target_price')
    y_test = test_df['target_price']
    logger.debug(f"Split the data into features and target")
    
    # Log dimensions of the features and target
    logger.debug(f"X_train: {X_train.shape}")
    logger.debug(f"y_train: {y_train.shape}")
    logger.debug(f"X_test: {X_test.shape}")
    logger.debug(f"y_test: {y_test.shape}")
    
    # build a model
    from src.models.current_price_baseline import CurrentPriceBaseLine
    
    model = CurrentPriceBaseLine()
    model.fit(X_train, y_train)
    logger.debug(f"Model built")
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.debug(f"Mean absolute error: {mae}")
    
    # push model to the model registry

if __name__ == "__main__":
    
    from src.config import config, hopsworks_config
    train_model(
        hopsworks_config=hopsworks_config,
        feature_view_name=config.feature_view_name,
        feature_view_version=config.feature_view_version,
        feature_group_name=config.feature_group_name,
        feature_group_version=config.feature_group_version,
        ohlc_window_sec=config.ohlc_window_sec,
        product_id=config.product_id,
        last_n_days=config.last_n_days,
        forecast_steps=config.forecast_steps,
    )
