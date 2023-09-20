import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import time

def initialize_mt5():
    if not mt5.initialize():
        print("Failed to initialize MT5.")
        return False
    return True

def get_real_time_data(symbol, timeframe, n_bars):
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    return pd.DataFrame(bars)

def preprocess_data(real_time_data):
    close_prices = np.array(real_time_data['close'])
    scaler = MinMaxScaler()
    close_prices = scaler.fit_transform(close_prices.reshape(-1, 1))
    nn_close_prices = close_prices.reshape(-1, 1)
    return nn_close_prices

def get_current_price(symbol):
    symbol_info = mt5.symbol_info_tick(symbol)
    if symbol_info is not None:
        return symbol_info.bid
    else:
        print(f"Failed to retrieve tick information for symbol: {symbol}")
        return None

def calculate_take_profit(predictions, pip_range, signal):
    current_price = get_current_price(symbol)
    if current_price is not None:
        if signal == "Buy":
            take_profit_price = current_price + pip_range * 0.0001
        elif signal == "Sell":
            take_profit_price = current_price - pip_range * 0.0001
        else:
            take_profit_price = None
        return take_profit_price
    else:
        print("Failed to retrieve current price.")
        return None

def generate_signals(symbol, timeframe, n_bars, model, aggregation_window, dynamic_threshold_factor, pip_range):
    position = None
    predictions_window = []
    
    while True:
        if not initialize_mt5():
            return
        
        real_time_data = get_real_time_data(symbol, timeframe, n_bars)
        nn_input = preprocess_data(real_time_data)
        predictions = model.predict(nn_input)
        predictions_window.append(predictions[-1])
        
        if len(predictions_window) > aggregation_window:
            predictions_window.pop(0)  # Remove the oldest prediction
        
        aggregated_signal = np.mean(predictions_window)
        buy_threshold = 0.5 + dynamic_threshold_factor
        sell_threshold = 0.5 - dynamic_threshold_factor
        
        if aggregated_signal > buy_threshold:
            signal = "Buy"
        elif aggregated_signal < sell_threshold:
            signal = "Sell"
        else:
            signal = "Hold"
        
        print(f"Signal: {signal}")
        
        if position is not None and ((position == "Buy" and signal == "Sell") or (position == "Sell" and signal == "Buy")):
            print("Closing the position.")
            position = None
        
        if position is None and (signal == "Buy" or signal == "Sell"):
            take_profit = calculate_take_profit(aggregated_signal, pip_range, signal)
            if take_profit is not None:
                print(f"Taking {signal} position. Take Profit Price: {take_profit}")
                position = signal
        
        mt5.shutdown()
        time.sleep(60)

if __name__ == "__main__":
    model = load_model("lstm_model.h5")
    symbol = "XAUUSDm"
    timeframe = mt5.TIMEFRAME_M1  # Higher Frequency Data (1-minute)
    n_bars = 500
    pip_range = 5  # Pip range, adjust as needed
    aggregation_window = 5  # Signal Aggregation (Last 5 minutes)
    dynamic_threshold_factor = 0.1  # Dynamic Thresholds
    generate_signals(symbol, timeframe, n_bars, model, aggregation_window, dynamic_threshold_factor, pip_range)
