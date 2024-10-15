import pandas as pd 


class MovingAverageBaseLine:
    """
    
    
    """
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Implementation required")