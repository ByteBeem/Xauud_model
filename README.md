# LSTM Model for XAU/USD Price Prediction

This repository contains a Long Short-Term Memory (LSTM) model for predicting the XAU/USD (Gold to US Dollar) price. The model is designed to analyze historical price data and provide forecasts for future price movements.

## Overview

- **Model Architecture**: This LSTM model utilizes recurrent neural networks (RNNs) to capture temporal dependencies in the price data, making it well-suited for time series forecasting.

- **Dataset**: The model is trained on historical XAU/USD price data, which is included in the `data` directory of this repository.

- **Evaluation**: We assess the model's performance using various evaluation metrics, such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

## Installation

To use this LSTM model, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/Bytebeem/Xauud_model.git
cd Xauud_model

pip install -r requirements.txt

## Usage
To train the LSTM model and make predictions, you can run the following command:

python live_trading.py

## Evaluation
To assess the model's performance, we calculate the following metrics:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
