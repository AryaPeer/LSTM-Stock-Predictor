import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

class Config:
    hidden_size = 64  # LSTM hidden size, reduced for simpler learning
    lstm_layers = 2    # Number of LSTM layers
    dropout_rate = 0.2  # Reduced Dropout rate to avoid losing too much information
    time_step = 90      # Time step adjusted slightly higher
    batch_size = 32     # Lowered batch size for finer updates
    learning_rate = 0.001
    epoch = 50          # More epochs for better training
    valid_data_rate = 0.2
    random_seed = 42

config = Config()

# Function to load stock data
def load_stock_data(stock_ticker):
    end = pd.Timestamp.now().strftime('%Y-%m-%d')
    start = '2010-01-01'  # Start from a long range for optimal range selection
    data = yf.download(stock_ticker, start=start, end=end)
    return data[['Close']]

# Preprocess the stock data for LSTM
def preprocess_data(data):
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scaling to range (0, 1)
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data, scaler

# Build and train the LSTM model
def build_and_train_model(scaled_data, training_data_len, config):
    train_data = scaled_data[0:training_data_len, :]
    
    x_train = []
    y_train = []
    
    for i in range(config.time_step, len(train_data)):
        x_train.append(train_data[i - config.time_step:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Correct input shape for LSTM
    
    # Build LSTM model with reduced dropout
    model = Sequential()
    model.add(LSTM(config.hidden_size, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(config.dropout_rate))
    
    if config.lstm_layers > 1:
        model.add(LSTM(config.hidden_size, return_sequences=False))
        model.add(Dropout(config.dropout_rate))
    
    model.add(Dense(1))  # Output layer predicting 1 value
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model with validation split and early stopping
    history = model.fit(x_train, y_train, batch_size=config.batch_size, epochs=config.epoch, validation_split=config.valid_data_rate, callbacks=[early_stop])
    
    # Plot training & validation loss
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    return model

# Predict stock prices using the trained LSTM model
def predict_stock_prices(model, scaled_data, training_data_len, scaler, config):
    test_data = scaled_data[training_data_len - config.time_step:, :]
    
    x_test = []
    
    for i in range(config.time_step, len(test_data)):
        x_test.append(test_data[i - config.time_step:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Correct shape for LSTM input
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)  # Convert back to original scale
    
    return predictions

# Plot the predicted prices against the actual stock prices
def plot_predictions(data, predictions, training_data_len):
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    
    plt.figure(figsize=(16, 6))
    plt.title('Model Predictions vs Real Prices')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'], label='Train Data')
    plt.plot(valid['Close'], label='Actual Prices')
    plt.plot(valid['Predictions'], label='Predicted Prices')
    plt.legend()
    plt.show()

# Full workflow to predict stock prices using LSTM
def predict_stock(stock_ticker):
    data = load_stock_data(stock_ticker)
    
    scaled_data, scaler = preprocess_data(data)
    
    training_data_len = int(len(scaled_data) * (1 - config.valid_data_rate))  # 80% train, 20% validation
    
    model = build_and_train_model(scaled_data, training_data_len, config)
    
    predictions = predict_stock_prices(model, scaled_data, training_data_len, scaler, config)
    
    plot_predictions(data, predictions, training_data_len)

# Example usage:
predict_stock('AAPL')
