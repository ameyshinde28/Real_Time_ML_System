import pandas as pd 


class CurrentPriceBaseLine:
    """
    
    
    """
    
    def __init__(self):
        pass 
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        """
        pass       
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        

        Args:
            X (pd.DataFrame): _description_

        Returns:
            pd.Series: _description_
        """
        return X['close']