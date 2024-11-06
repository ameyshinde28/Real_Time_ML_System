from typing import Optional

from loguru import logger
from xgboost import XGBRegressor
import pandas as pd
import optuna

class XGBoostModel:
    def __init__(self):
        # self.model = XGBRegressor()
        self.model = None
        
    def fit(
            self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series,
            n_search_trials: Optional[int] = 10,
            n_splits: Optional[int] = 3,
            ):
        """
        trains an XGBoost model on a given training data.

        Args:
            X_train (pd.DataFrame): _description_
            y_train (pd.Series): _description_
            n_search_trails (Optional[int], optional): _description_. Defaults to 0.
            n_splits (Optional[int], optional): _description_. Defaults to 3.
        """
        logger.info(f"Training XGBoost model with n_search_trials={n_search_trials} and n_splits={n_splits}")
        
        assert n_search_trials >= 0, "n_search_trials must be non-negative"
        if n_search_trials == 0:
            # Train a model with default parameters 
            # This is what we have been using so for
            self.model = XGBRegressor()    
            self.model.fit(X_train, y_train)
            logger.info(f"Model with default hyperparameters")
        else:
            # We do cross-validation with the number of splits specified
            # and we search for the best hyperparameters using Bayesian optimization
            best_hyperparams = self._find_best_hyperparameters(X_train=X_train, 
                                                               y_train=y_train,
                                                               n_search_trials=n_search_trials,
                                                               n_splits=n_splits)
            logger.info(f"Best hyperparameters: {best_hyperparams}")
            self.model = XGBRegressor(**best_hyperparams) 
            self.model.fit(X_train, y_train)
            logger.info(f"Model trained with Best hyperparameters")
            
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def _find_best_hyperparameters(self,
                                   X_train: pd.DataFrame,
                                   y_train: pd.Series,
                                   n_search_trials: int,
                                   n_splits:int,):   
        """
        FInd the best hyperparameters
        Args:
            X_train (pd.DataFrame): _description_
            y_train (pd.Series): _description_
            n_search_trials (int): _description_
            n_splits (int): _description_

        Returns:
            _type_: _description_
        """
        def objective(trial: optuna.Trial) -> float:
            """
            Objestive function for Optuna that returns the mean absolute error we
            want to minimize.
            Args:
                trial (optuna.Trial): _description_

            Returns:
                float: _description_
            """
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("subsample", 0.5, 1.0) 
            }
            # Let's split our X_train into n_splits folds with a time-series split
            # we want to keep the timeseries order in each fold
            # we will ue the time series split from sklearn
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import mean_absolute_error
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            mae_scores = []
            # splitting the data into training and validation sets
            for train_index, test_index in tscv.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
                y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
                
                # train the model on the validation set
                model = XGBRegressor(**params)
                model.fit(X_train_fold, y_train_fold)
                
                
                # evalute the model on the validation set
                y_pred = model.predict(X_val_fold)
                mae = mean_absolute_error(y_val_fold, y_pred)
                mae_scores.append(mae)
            
            # return the average MAE across all folds 
            import numpy as np
            return np.mean(mae_scores)
    
        # We Create  a study object that minimizes the objective function
        study = optuna.create_study(direction="minimize")
        
        # We run the trails
        study.optimize(objective, n_trials=n_search_trials)
        
        # we return the best hyperparameters
        return study.best_trial.params
    
    
    def get_model_obj(self):
        return self.model
