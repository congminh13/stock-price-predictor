import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

class ImprovedSPPModule:
  def __init__(self, stock_symbol, period, train_ratio):
    self.stock_symbol = stock_symbol
    self.period = period
    self.train_ratio = train_ratio
    self.model = LinearRegression()
    self.data = None
    self.X_train = None
    self.X_test = None
    self.y_train = None
    self.y_test = None
    self.y_pred = None

  def fetch_data(self):
    try:
      self.data = yf.download(self.stock_symbol, period=self.period, progress=False)
      if self.data.empty:
        raise ValueError("No data fetched. Please check the stock symbol or period.")
      self.data['Days'] = np.arange(len(self.data))
      return self.data
    except Exception as e:
      st.error(f"Error fetching data: {e}")
      return None


  @staticmethod
  def calculate_rsi(data, period=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

  @staticmethod
  def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

  def prepare_data(self):
    self.data['RSI'] = self.calculate_rsi(self.data)
    self.data['MACD'], self.data['Signal Line'] = self.calculate_macd(self.data)

    self.data.dropna(inplace=True)

    X = self.data[['Days', 'RSI', 'MACD']]
    y = self.data['Close']

    train_days = int(len(self.data) * self.train_ratio)
    self.X_train = X[:train_days]
    self.X_test = X[train_days:]
    self.y_train = y[:train_days]
    self.y_test = y[train_days:]
    return self.X_train, self.X_test, self.y_train, self.y_test

  def train_model(self):
    self.model.fit(self.X_train, self.y_train)

  def predict(self):
    self.y_pred = self.model.predict(self.X_test)

  def evaluate(self):
    mse = mean_squared_error(self.y_test, self.y_pred)
    mae = mean_absolute_error(self.y_test, self.y_pred)
    r2 = r2_score(self.y_test, self.y_pred)
    return {"MSE": mse, "MAE": mae, "R2": r2}

  def visualize(self):
    plt.figure(figsize=(14, 8))
    plt.plot(self.data['Days'], self.data['Close'], label="Actual Prices", color="blue", marker="o")
    plt.plot(self.X_train['Days'], self.model.predict(self.X_train), label="Predicted Prices (Training)", color="green", linestyle="--")
    plt.plot(self.X_test['Days'], self.y_pred, label="Predicted Prices", color="purple", linestyle="--")
    plt.axvline(x=self.X_test['Days'].iloc[0], color="grey", linestyle="--", label="Train-Test Split")
    plt.xlabel("Days")
    plt.ylabel("Stock Closing Price")
    plt.title(f"Stock Price Prediction for {self.stock_symbol}")
    plt.legend()
    plt.grid()
    st.pyplot(plt)
    metrics = self.evaluate()
    st.write(f"- Mean Squared Error (MSE): {metrics['MSE']:.2f}")
    st.write(f"- Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    st.write(f"- R-squared (R^2): {metrics['R2']:.2f}")
    plt.close()