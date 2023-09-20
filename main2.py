import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tvDatafeed import TvDatafeed, Interval
import os

# Function to calculate Exponential Moving Average (EMA)
def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(data, window):
    price_diff = data.diff()
    gain = np.where(price_diff > 0, price_diff, 0)
    loss = np.where(price_diff < 0, -price_diff, 0)
    avg_gain = calculate_ema(pd.Series(gain), window)
    avg_loss = calculate_ema(pd.Series(loss), window)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Moving Average Convergence Divergence (MACD)
def calculate_macd(data, short_window, long_window, signal_window):
    short_ema = calculate_ema(data, short_window)
    long_ema = calculate_ema(data, long_window)
    macd_line = short_ema - long_ema
    signal_line = calculate_ema(macd_line, signal_window)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window, num_std):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    return upper_band, lower_band

# Function to generate support and resistance breakout signals
def generate_breakout_signals(data, support_level, resistance_level):
    data["Support_Breakout"] = np.where(data["Low"] < support_level, 1, 0)
    data["Resistance_Breakout"] = np.where(data["High"] > resistance_level, 1, 0)
    return data

# Define constants
USERNAME = 'dony041209'
PASSWORD = 'DonaldRSA04?'

# Initialize TV datafeed
tv = TvDatafeed(USERNAME, PASSWORD)

# Fetch historical price data for XAUUSD using TradingView datafeed
symbol = 'OANDA:XAUUSD'
interval = Interval.in_5_minute
n_bars = 5000

# Fetch historical data
new_data = tv.get_hist(symbol=symbol, interval=interval, n_bars=n_bars)

# Create DataFrame from the fetched data
data = pd.DataFrame({
    'Open': new_data.open,
    'High': new_data.high,
    'Low': new_data.low,
    'Close': new_data.close,
    'Volume': new_data.volume
})

# Calculate price movement: 1 for bullish, 0 for bearish
data["Price_Diff"] = data["Close"].diff()
data["Price_Movement"] = np.where(data["Price_Diff"] > 0, 1, 0)

# Add technical indicators as features
data["EMA_10"] = calculate_ema(data["Close"], window=10)
data["RSI_14"] = calculate_rsi(data["Close"], window=14)
data["MACD_Line"], _, _ = calculate_macd(data["Close"], short_window=12, long_window=26, signal_window=9)
data["BB_Upper"], data["BB_Lower"] = calculate_bollinger_bands(data["Close"], window=20, num_std=2)

# Define support and resistance levels
support_level = 150
resistance_level = 200

# Generate support and resistance breakout signals
data = generate_breakout_signals(data, support_level, resistance_level)

# Data preprocessing
close_prices = np.array(new_data['close'])
scaler = MinMaxScaler()
close_prices = scaler.fit_transform(close_prices.reshape(-1, 1))
nn_close_prices = close_prices.reshape(-1, 1)

# Shift the close prices to get next close prices
nn_next_close_prices = np.roll(nn_close_prices, -1)

# Remove last entry to match the reshaped nn_close_prices
nn_close_prices = nn_close_prices[:-1]
nn_next_close_prices = nn_next_close_prices[:-1]

# Reshape the data
nn_close_prices = nn_close_prices.reshape(-1, 1, 1)
nn_next_close_prices = nn_next_close_prices.reshape(-1, 1, 1)

# Split data into training and validation sets
nn_train_data, nn_val_data, nn_train_labels, nn_val_labels = train_test_split(
    nn_close_prices, nn_next_close_prices, test_size=0.2, random_state=42)

# Initialize model
model = None

while True:
    # Check if the model file exists, and load if it does
    if model is None and "lstm_model.h5" in os.listdir():
        model = load_model("lstm_model.h5")
    else:
        model = Sequential([
            LSTM(50, input_shape=(nn_train_data.shape[1], nn_train_data.shape[2]), activation='relu', return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    # Train the model
    lstm_history = model.fit(
        nn_train_data, nn_train_labels, validation_data=(nn_val_data, nn_val_labels),
        epochs=200,  batch_size=32, verbose=2)

    # Evaluate the model
    mse = model.evaluate(nn_val_data, nn_val_labels)[1]
    print(f"Mean Squared Error: {mse:.2f}")

    # Save the model
    model.save("lstm_model.h5")

    # Predictions
    y_pred = model.predict(nn_val_data)

    # Reshape nn_val_labels to match the expected shape
    nn_val_labels_reshaped = nn_val_labels.reshape(-1, 1)

    # Plot predicted vs. actual movements
    plt.figure(figsize=(12, 6))
    plt.plot(nn_val_labels_reshaped, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linestyle='dashed', linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Price Movement")
    plt.title("Actual vs. Predicted Price Movement")
    plt.legend()
    plt.grid(True)
    plt.show()
