import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
import matplotlib.pyplot as plt

class KerasSPPModule:
    def __init__(self, stock_symbol, period, train_ratio, look_back=60):
        self.stock_symbol = stock_symbol
        self.period = period
        self.train_ratio = train_ratio
        self.look_back = look_back
        self.model = None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_train = None
        self.scaler_y = MinMaxScaler()

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

    def prepare_data(self):
        if self.data is None:
            self.fetch_data()
        if self.data is None or self.data.empty:
            st.error("No data available to prepare.")
            return None, None, None, None

        self.data.dropna(inplace=True)

        if len(self.data) < 10:
            st.error(f"Not enough data points ({len(self.data)}) to prepare sequences.")
            return None, None, None, None

        self.look_back = min(self.look_back, len(self.data) // 2)
        # st.write(f"Adjusted look_back to {self.look_back} due to data size ({len(self.data)}).")

        # Features: Close only
        target = self.data['Close'].values.reshape(-1, 1)

        # Normalize target
        scaled_target = self.scaler_y.fit_transform(target)

        # Create sequences for LSTM
        X, y = [], []
        for i in range(self.look_back, len(self.data)):
            X.append(scaled_target[i - self.look_back:i])
            y.append(scaled_target[i, 0])
        X, y = np.array(X), np.array(y)

        # st.write(f"X shape: {X.shape}, y shape: {y.shape}")

        if X.shape[0] == 0:
            st.error("No sequences generated. Data may be insufficient.")
            return None, None, None, None

        train_days = int(len(X) * self.train_ratio)
        if train_days < 1 or len(X) - train_days < 1:
            st.error("Not enough data for training and testing split.")
            return None, None, None, None

        self.X_train = X[:train_days]
        self.X_test = X[train_days:]
        self.y_train = y[:train_days]
        self.y_test = y[train_days:]

        # st.write(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        # st.write(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rs = rs.replace([np.inf, -np.inf], np.nan) 
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
        short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal_line

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.look_back, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self):
        if self.X_train is None or self.y_train is None:
            st.error("Training data not prepared. Call prepare_data first.")
            return
        self.build_model()
        self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, verbose=0)

    def predict(self):
        if self.X_test is None or self.model is None:
            st.error("Cannot predict. Ensure model is trained and test data is prepared.")
            return
        # Test predictions
        self.y_pred = self.model.predict(self.X_test, verbose=0)
        self.y_pred = self.scaler_y.inverse_transform(self.y_pred)
        self.y_test = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1))
        # Training predictions
        self.y_pred_train = self.model.predict(self.X_train, verbose=0)
        self.y_pred_train = self.scaler_y.inverse_transform(self.y_pred_train)
        self.y_train = self.scaler_y.inverse_transform(self.y_train.reshape(-1, 1))

    def evaluate(self):
        if self.y_test is None or self.y_pred is None:
            return {"MSE": float('nan'), "MAE": float('nan'), "R2": float('nan')}
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        return {"MSE": mse, "MAE": mae, "R2": r2}

    def visualize(self):
        if self.y_test is None or self.y_pred is None or self.y_pred_train is None:
            st.error("Cannot visualize. Ensure predictions are made.")
            return
        plt.figure(figsize=(14, 6))
        # Plot actual prices
        plt.plot(self.data['Days'][self.look_back:], self.data['Close'][self.look_back:], label="Actual Prices", color="blue", marker="o")
        # Plot training predictions
        train_days = self.data['Days'][self.look_back:self.look_back + len(self.y_pred_train)]
        plt.plot(train_days, self.y_pred_train, label="Keras Model (Train)", color="purple", linestyle="--")
        # Plot test predictions
        test_days = self.data['Days'][self.look_back + len(self.y_pred_train):]
        plt.plot(test_days, self.y_pred, label="Keras Model (Predict)", color="cyan", linestyle="--")
        # Train-test split line
        split_day = self.data['Days'][self.look_back + len(self.y_pred_train)]
        plt.axvline(x=split_day, color="grey", linestyle="--", label="Train-Test Split")
        plt.xlabel("Days")
        plt.ylabel("Stock Closing Price")
        plt.title(f"Keras Stock Price Prediction for {self.stock_symbol}")
        plt.legend()
        plt.grid()
        st.pyplot(plt)
        metrics = self.evaluate()
        st.write(f"- Mean Squared Error (MSE): {metrics['MSE']:.2f}")
        st.write(f"- Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
        st.write(f"- R-squared (R^2): {metrics['R2']:.2f}")
        plt.close()

    def predict_future(self, start_date, end_date):
        if self.data is None or self.model is None:
            st.error("Model or data not initialized.")
            return None
        future_data = yf.download(self.stock_symbol, start=self.data.index[-1], end=end_date, progress=False)
        if future_data.empty:
            st.error("No future data available for the specified date range.")
            return None

        future_data.dropna(inplace=True)

        if len(future_data) < self.look_back:
            st.error(f"Not enough future data points ({len(future_data)}) for look_back period ({self.look_back}).")
            return None

        features = future_data[['Close']].values
        scaled_features = self.scaler_y.transform(features)

        X_future = []
        for i in range(self.look_back, len(future_data)):
            X_future.append(scaled_features[i - self.look_back:i])
        X_future = np.array(X_future)

        if X_future.shape[0] == 0:
            st.error("No future sequences generated.")
            return None

        future_pred = self.model.predict(X_future, verbose=0)
        future_pred = self.scaler_y.inverse_transform(future_pred)

        future_dates = future_data.index[self.look_back:]
        result_df = pd.DataFrame({
            'Date': future_dates[:len(future_pred)],
            'Predicted Close': future_pred.flatten()
        })

        return result_df