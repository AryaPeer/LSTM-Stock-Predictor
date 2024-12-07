import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Attention, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import ta 
from sklearn.model_selection import TimeSeriesSplit

class Config:
    hidden_size = 64  
    lstm_layers = 2    
    dropout_rate = 0.2  
    time_step = 90     
    batch_size = 32     
    learning_rate = 0.001
    epoch = 25     
    valid_data_rate = 0.2
    random_seed = 42

config = Config()

np.random.seed(config.random_seed)
tf.random.set_seed(config.random_seed)

def load_stock_data(stock_ticker):
    full_data = yf.Ticker(stock_ticker).history(period='max')

   
    start_date = '2010-01-01'
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    data = full_data[(full_data.index >= start_date) & (full_data.index <= end_date)]

    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(close=data['Close']).macd()

    data.dropna(inplace=True)

    data['DayOfWeek'] = data.index.dayofweek
    data['DayOfMonth'] = data.index.day
    data['Month'] = data.index.month

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'EMA_20', 'RSI', 'MACD', 'DayOfWeek', 'DayOfMonth', 'Month']
    return data[features]

def preprocess_data(data, future_steps):
    dataset = data.values
    train_data = dataset[:-future_steps]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    scaled_data = scaler.transform(dataset)
    return scaled_data, scaler

def prepare_multistep_data(scaled_data, config, future_steps):
    x_train = []
    y_train = []
    num_features = scaled_data.shape[1]

    for i in range(config.time_step, len(scaled_data) - future_steps + 1):
        x_train.append(scaled_data[i - config.time_step:i])
        y_train.append(scaled_data[i:i + future_steps, 3]) 

    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train

def build_and_train_model(scaled_data, config, future_steps):
    x_train, y_train = prepare_multistep_data(scaled_data, config, future_steps)
    num_features = x_train.shape[2]

    input_layer = Input(shape=(config.time_step, num_features))
    x = Bidirectional(LSTM(config.hidden_size, return_sequences=True))(input_layer)
    x = Dropout(config.dropout_rate)(x)
    x = Bidirectional(LSTM(config.hidden_size, return_sequences=True))(x)
    x = Dropout(config.dropout_rate)(x)

    attention = Attention()([x, x])
    attention_flat = Flatten()(attention)
    output = Dense(future_steps)(attention_flat)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate), loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        x_train,
        y_train,
        batch_size=config.batch_size,
        epochs=config.epoch,
        validation_split=config.valid_data_rate,
        callbacks=[early_stop],
        verbose=1
    )

    return model

def predict_future_prices(model, last_sequence, scaler, future_steps, last_actual_price):
    X = np.array([last_sequence])

    pred_scaled = model.predict(X)

    pred_full = np.zeros((future_steps, scaler.scale_.shape[0]))

    pred_full[:, 3] = pred_scaled[0]  


    predictions = scaler.inverse_transform(pred_full)[:, 3]  

    adjustment = last_actual_price - predictions[0]
    adjusted_predictions = predictions + adjustment

    return adjusted_predictions

def plot_predictions(data, predictions, days_to_predict):
    last_six_months = data.last('6M')
    
    last_date = data.index[-1]
    prediction_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days_to_predict,
        freq='B' 
    )
    prediction_df = pd.DataFrame(
        data=predictions,
        index=prediction_dates,
        columns=['Predictions']
    )
    
    combined_data = pd.concat([last_six_months['Close'], prediction_df['Predictions']])
    
    plt.figure(figsize=(16, 6))
    plt.title(f'Model Predictions for Next {days_to_predict} Days')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(combined_data.index, combined_data.values, label='Historical and Predicted Prices')
    plt.axvline(x=last_date, color='r', linestyle='--', label='Prediction Start')
    plt.legend()

    plot_path = "temp_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


def predict_stock(stock_ticker, days_to_predict=60):
    data = load_stock_data(stock_ticker)
    future_steps = days_to_predict
    scaled_data, scaler = preprocess_data(data, future_steps)

    model = build_and_train_model(scaled_data, config, future_steps)

    last_sequence = scaled_data[-config.time_step:]

    last_actual_price = data['Close'].values[-1]

    predictions = predict_future_prices(model, last_sequence, scaler, future_steps, last_actual_price)
    plot_path = plot_predictions(data, predictions, days_to_predict)

    return plot_path

# Example usage:
# predict_stock('AAPL')