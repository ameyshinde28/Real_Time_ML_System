

def get_model_name(
    product_id: str,
    ohlc_window_sec: int,
    forecast_steps: int,

) -> str:
    """
    returns the name of the model in the model registry given the
    -product_id
    -ohlc_window_sec
    -forecast_steps

    Args:
        product_id (str): _description_
        ohlc_window_sec (int): _description_
        forecast_steps (int): _description_

    Returns:
        str: _description_
    """
    return f"price_predictor_{product_id.replace('/', '_')}_{ohlc_window_sec}s_{forecast_steps}steps"