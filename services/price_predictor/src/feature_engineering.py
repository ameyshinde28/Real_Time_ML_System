import pandas as pd
import talib


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.
    
    This function calculates and adds Simple Moving Averages (SMA) for 5, 10, and 20 periods
    to the input DataFrame. It uses the 'close' price for SMA calculations.

    Args:
        df (pd.DataFrame): Input DataFrame with the following required columns:
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - close: Closing price
            - volume: Trading volume

    Returns:
        pd.DataFrame: A new DataFrame with the original data and the following additional columns:
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - close: Closing price
            - volume: Trading volume
            - 'XXX'

    Raises:
        ValueError: If the input DataFrame does not contain a 'close' column.

    Note:
        - This function uses TA-Lib for calculating SMAs.
        - NaN values created during SMA calculation are removed from the resulting DataFrame.
        - The first few rows of the DataFrame may be dropped due to NaN values in SMA calculations.
    """
    # Ensure the DataFrame has a 'close' column
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    # Add Simple Moving Average (SMA) for different periods
    df['SMA_7'] = talib.SMA(df['close'], timeperiod=7)
    df['SMA_14'] = talib.SMA(df['close'], timeperiod=14)
    df['SMA_28'] = talib.SMA(df['close'], timeperiod=28)

    # Exponential Moving Averages (EMA)
    df['EMA_7'] = talib.EMA(df['close'], timeperiod=7)
    df['EMA_14'] = talib.EMA(df['close'], timeperiod=14)
    df['EMA_28'] = talib.EMA(df['close'], timeperiod=28)

    # Moving Average Convergence Divergence (MACD)
    macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    
    # Bollinger bands
    upper, lower, middle = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BB_Upper']= upper
    df['BB_Middle']= middle
    df['BB_Lower']=lower

    #  Relative Strength Index (RSI)
    df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)

    # Stochastic Oscillator
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['Stoch_K'] = slowk
    df['Stoch_D'] = slowd

    # OBV
    df['OBV'] = talib.OBV(df['close'], df['volume'])

    # ATR
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # CCI
    
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Chaikin Money Flow (CMF)
    df['CMF'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
    
    # Remove NaN values created by the SMA calculation
    # df.dropna(inplace=True)

    return df
