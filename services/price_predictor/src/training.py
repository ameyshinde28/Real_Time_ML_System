from typing import Optional

from comet_ml import Experiment
from sklearn.metrics import mean_absolute_error
from loguru import logger
import joblib
import os

from src.config import HopsworksConfig, CometConfig
from src.feature_engineering import (
    add_technical_indicators,
    add_temporal_features,
    )
from src.models.current_price_baseline import CurrentPriceBaseLine
from src.models.xgboost_model import XGBoostModel
from src.utils import hash_dataframe


def train_model(
    comet_config: CometConfig,
    feature_view_name: str,
    feature_view_version: int,
    feature_group_name: str,
    feature_group_version: int,
    ohlc_window_sec: int,
    product_id: str,
    last_n_days:int,
    hopsworks_config: HopsworksConfig,
    forecast_steps: int,
    prec_test_data: Optional[float] = 0.3,
    n_search_trials: Optional[int] = 10,
    n_splits: Optional[int] = 3,
    
):
    """_summary_

    Args:
        comet_config (CometConfig): _description_
        feature_view_name (str): _description_
        feature_view_version (int): _description_
        feature_group_name (str): _description_
        feature_group_version (int): _description_
        ohlc_window_sec (int): _description_
        product_id (str): _description_
        last_n_days (int): _description_
        hopsworks_config (HopsworksConfig): _description_
        forecast_steps (int): _description_
        prec_test_data (Optional[float], optional): _description_. Defaults to 0.3.
        n_search_trials (Optional[int], optional): _description_. Defaults to 10.
        n_splits (Optional[int], optional): _description_. Defaults to 3.
    """
    
    # Create a comet experiment
    experiment =Experiment(
        api_key=comet_config.comet_ml_api_key,
        project_name=comet_config.comet_project_name,
    )
    
    experiment.log_parameter("last_n_days", last_n_days)
    experiment.log_parameter("forecast_steps", forecast_steps)
    experiment.log_parameter("n_search_trials", n_search_trials)
    experiment.log_parameter("n_splits", n_splits)
    
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
    experiment.log_parameter("n_raw_feature_rows", len(ohlc_data))
    
    # log a hsah of the dataset to comet
    dataset_hash = hash_dataframe(ohlc_data)
    experiment.log_parameter("ohlc_data_hash", dataset_hash)
    
    # split the data into training and testing
    logger.debug(f"Splitting the data into training and testing")
    test_size = int(len(ohlc_data) * prec_test_data)
    train_df = ohlc_data[:-test_size]
    test_df = ohlc_data[-test_size:]
    logger.debug(f"Training data: {len(train_df)}")
    logger.debug(f"Testing data: {len(test_df)}")
    experiment.log_parameter("n_train_rows_before_dropna", len(train_df))
    experiment.log_parameter("n_test_rows_before_dropna", len(test_df))

    
    
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
    experiment.log_parameter("n_train_rows_after_dropna", len(train_df))
    experiment.log_parameter("n_test_rows_after_dropna", len(test_df))

    
    # Split the data into features and target
    X_train = train_df.drop(columns='target_price')
    y_train = train_df['target_price']
    X_test = test_df.drop(columns='target_price')
    y_test = test_df['target_price']
    logger.debug(f"Split the data into features and target")
    
    # Keep only the features that are needed for the model
    
    X_train = X_train[['open', 'high', 'low', 'close', 'volume']]
    X_test = X_test[['open', 'high', 'low', 'close', 'volume']]
    
    
    # Log dimensions of the features and target
    logger.debug(f"X_train: {X_train.shape}")
    logger.debug(f"y_train: {y_train.shape}")
    logger.debug(f"X_test: {X_test.shape}")
    logger.debug(f"y_test: {y_test.shape}")
    
    # add technical indicator to the features
    X_train = add_technical_indicators(X_train)
    X_test = add_technical_indicators(X_test)
    logger.debug(f"Add technical indicator to the features")
    logger.debug(f"X_train: {X_train.columns}")
    logger.debug(f"X_test: {X_test.columns}")
    experiment.log_parameter("features", X_train.columns.tolist())
    
    # add temporal featuress from the timestamp_ms column
    X_train = add_temporal_features(X_train)
    X_test = add_temporal_features(X_test)
    logger.debug(f"Add temporal_features to the features from the timestamp_ms column")
    logger.debug(f"X_train: {X_train.columns}")
    logger.debug(f"X_test: {X_test.columns}")
    experiment.log_parameter("features", X_train.columns.tolist())
    experiment.log_parameter("n_features", len(X_train.columns))


    # Dropping rows with NaN values
    # Extract nan row train
    nan_rows_train = X_train.isna().any(axis=1)
    # Count nan rows train
    logger.debug(f"Number of Nan rows in X_train: {nan_rows_train.sum()}")
    # Keep only non nan rows train
    X_train = X_train.loc[~nan_rows_train]
    y_train = y_train.loc[~nan_rows_train]
    
    # Extract nan rows test
    nan_rows_test = X_test.isna().any(axis=1)
    # Count nan rows test
    logger.debug(f"Number of Nan rows in X_train: {nan_rows_test.sum()}")
    # Keep only non nan rows test
    X_test = X_test.loc[~nan_rows_test]
    y_test = y_test.loc[~nan_rows_test]
 
    # Percentage of dropped rows
    experiment.log_parameter("n_nan_rows_train", nan_rows_train.sum())
    experiment.log_parameter("n_nan_rows_test", nan_rows_test.sum())
    experiment.log_parameter("perc_dropped_rows_train", nan_rows_train.sum() / len(X_train) * 100)
    experiment.log_parameter("perc_dropped_rows_test", nan_rows_test.sum() / len(X_test) * 100)
    
    # X_train=X_train.dropna()
    # X_test=X_test.dropna()
    
    # Log dimensions of the features and target
    logger.debug(f"X_train: {X_train.shape}")
    logger.debug(f"y_train: {y_train.shape}")
    logger.debug(f"X_test: {X_test.shape}")
    logger.debug(f"y_test: {y_test.shape}")

    # Log dimensions of the features and target to Comet ML
    experiment.log_parameter("X_train:", X_train.shape)
    experiment.log_parameter("y_train:", y_train.shape)
    experiment.log_parameter("X_test:", X_test.shape)
    experiment.log_parameter("y_test:", y_test.shape)
    
    # breakpoint()
    
    # build a model
    
    
    model = CurrentPriceBaseLine()
    model.fit(X_train, y_train)
    logger.debug(f"Model built")
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.debug(f"Mean absolute error: {mae}")
    experiment.log_metric("MAE_CurrentPriceBaseline", mae)
    mae_baseline = mae
    
    # compute mae on the training  data for debugging purpose
    y_train_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    logger.debug(f"Mean absolute error on the training data of CurrentPriceBaseline: {mae_train}")
    experiment.log_metric("mae_training_CurrentPriceBaseline", mae_train)
    
    # breakpoint()
    # train an XGBoost model
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train, n_search_trials=n_search_trials, n_splits=n_splits)
    y_pred = xgb_model.predict(X_test)
    
    # Compute on test data
    mae = mean_absolute_error(y_test, y_pred)
    logger.debug(f"Mean Absolute Error: {mae}")
    experiment.log_metric("mae_XGBRegressor", mae)
    
    # compute mae on the training  data for debugging purpose
    y_train_pred = xgb_model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    logger.debug(f"Mean absolute error on the training data of XGBRegressor model: {mae_train}")
    experiment.log_metric("mae_training_XGBRegressor model", mae_train)
    
    
 
    # push model to the model registry
    
    model_name = f"price_predictor_{product_id.replace('/','_')}_{ohlc_window_sec}s_{forecast_steps}steps"
    local_model_path = f"{model_name}.joblib"
    joblib.dump(xgb_model.get_model_obj(), local_model_path)
   
   # Log the model  to comet ml
    experiment.log_model(
        name=model_name,
        file_or_folder=local_model_path,
        overwrite=True,
        # model_framework="xgboost",
        # model_format="joblib"
    )
    
    if mae < mae_baseline:
        logger.info(f"Model {model_name} is better than the baseline model. Pushing to Model Registry")
        # Register the model in  Comet ML
        registered_model = experiment.register_model(
            model_name=model_name,
            # overwrite=True,
            
        )
    else:
        logger.info(f"Model {model_name} is not better than the baseline model. Not pushing to Model Registry")
    
    # Clean up the local model file
    os.remove(local_model_path)
    
    experiment.end()

if __name__ == "__main__":
    
    from src.config import config, hopsworks_config, comet_config
    train_model(
        comet_config=comet_config,
        hopsworks_config=hopsworks_config,
        feature_view_name=config.feature_view_name,
        feature_view_version=config.feature_view_version,
        feature_group_name=config.feature_group_name,
        feature_group_version=config.feature_group_version,
        ohlc_window_sec=config.ohlc_window_sec,
        product_id=config.product_id,
        last_n_days=config.last_n_days,
        forecast_steps=config.forecast_steps,
        n_search_trials=config.n_search_trials,
        n_splits=config.n_splits,
    )
