import pytest
from src.main import update_ohlcv_candle

def test_update_ohlcv_candle_higher_price():
    candle = {'open': 100, 'high': 110, 'low': 90, 'close': 105, 'volume': 10, 'product_id': 'BTC-USD'}
    trade = {'price': 115, 'quantity': 5, 'product_id': 'BTC-USD'}
    
    updated_candle = update_ohlcv_candle(candle, trade)
    
    assert updated_candle['high'] == 115
    assert updated_candle['close'] == 115
    assert updated_candle['volume'] == 15
    assert updated_candle['product_id'] == 'BTC-USD'

def test_update_ohlcv_candle_lower_price():
    candle = {'open': 100, 'high': 110, 'low': 90, 'close': 105, 'volume': 10, 'product_id': 'BTC-USD'}
    trade = {'price': 85, 'quantity': 3, 'product_id': 'BTC-USD'}
    
    updated_candle = update_ohlcv_candle(candle, trade)
    
    assert updated_candle['low'] == 85
    assert updated_candle['close'] == 85
    assert updated_candle['volume'] == 13
    assert updated_candle['product_id'] == 'BTC-USD'

def test_update_ohlcv_candle_same_price():
    candle = {'open': 100, 'high': 110, 'low': 90, 'close': 105, 'volume': 10, 'product_id': 'BTC-USD'}
    trade = {'price': 105, 'quantity': 2, 'product_id': 'BTC-USD'}
    
    updated_candle = update_ohlcv_candle(candle, trade)
    
    assert updated_candle['high'] == 110
    assert updated_candle['low'] == 90
    assert updated_candle['close'] == 105
    assert updated_candle['volume'] == 12
    assert updated_candle['product_id'] == 'BTC-USD'