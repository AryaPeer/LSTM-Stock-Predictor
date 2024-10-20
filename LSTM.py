import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

class StockPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.scaled_data = None
        self.scaler = None
        self.training_data_len = None
        self.model = None
        self.start = None
        self.end = None

    # Method to load stock data
    def load_stock_data(self, start='2010-01-01', end=None):
        end = end or pd.Timestamp.now().strftime('%Y-%m-%d')  # Default end date is today
        self.data = yf.download(self.ticker, start=start, end=end)[['Close']]
    
    # Method to preprocess data
    def preprocess_data(self):
        dataset = self.data.values
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(dataset)
        self.training_data_len = int(len(self.scaled_data) * 0.8)
    
    # Method to build and train the LSTM model
    def build_and_train_model(self):
        train_data = self.scaled_data[0:self.training_data_len, :]
        x_train, y_train = [], []
        
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Build LSTM model
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.model.add(LSTM(64, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        self.model.fit(x_train, y_train, batch_size=1, epochs=1, callbacks=[early_stop])
    
    # Method to predict stock prices
    def predict_prices(self):
        test_data = self.scaled_data[self.training_data_len - 60:, :]
        x_test, y_test = [], self.scaled_data[self.training_data_len:, :]
        
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        predictions = self.model.predict(x_test)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions, y_test
    
    # Method to calculate volatility
    def calculate_volatility(self):
        returns = self.data['Close'].pct_change().dropna()
        volatility = returns.rolling(window=30).std()  # 30-day rolling volatility
        return volatility
    
    # Method to select an optimal date range based on volatility
    def select_optimal_date_range(self, volatility_threshold=0.02):
        volatility = self.calculate_volatility()
        high_vol_period = self.data[volatility > volatility_threshold]
        
        if len(high_vol_period) < 60:
            self.start, self.end = self.data.index[0], self.data.index[-1]
        else:
            self.start, self.end = high_vol_period.index[0], high_vol_period.index[-1]
    
    # Method to plot predictions
    def plot_predictions(self, predictions):
        train = self.data[:self.training_data_len]
        valid = self.data[self.training_data_len:]
        valid['Predictions'] = predictions
        
        plt.figure(figsize=(16, 6))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()

    # Method to check for underfitting
    def check_for_underfitting(self, predictions, true_values):
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        return rmse > 0.05  # If RMSE > 5%, consider it underfitting
    
    # Method to start prediction with smart stopping and retraining
    def predict_with_smart_stopping(self):
        # Load initial stock data for default period (start='2010-01-01' and end='today')
        self.load_stock_data()
        
        # Select optimal date range based on volatility
        self.select_optimal_date_range()
        
        # Reload data with the selected start and end dates
        self.load_stock_data(start=self.start, end=self.end)
        
        # Preprocess and train model
        self.preprocess_data()
        self.build_and_train_model()
        
        # Get predictions
        predictions, y_test = self.predict_prices()
        
        # Check for underfitting
        if self.check_for_underfitting(predictions, y_test):
            print("Underfitting detected. Retraining with extended date range.")
            self.select_optimal_date_range(volatility_threshold=0.015)
            self.predict_with_smart_stopping()
        else:
            self.plot_predictions(predictions)

# Example usage:
# stock = StockPredictor('AAPL')
# stock.predict_with_smart_stopping()
