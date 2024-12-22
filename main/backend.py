import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Attention, Input, Flatten, Add
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import ta

class Config:
    hidden_size = 256
    lstm_layers = 2
    dropout_rate = 0.2  
    time_step = 30 
    batch_size = 32   
    learning_rate = 0.001
    epoch = 100
    valid_data_rate = 0.2
    random_seed = 42

config = Config()

np.random.seed(config.random_seed)
tf.random.set_seed(config.random_seed)

def load_stock_data(stock_ticker):
    full_data = yf.Ticker(stock_ticker).history(period='max')
    
    start_date = '2010-01-01'
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    data = full_data.loc[
        (full_data.index >= start_date) & 
        (full_data.index <= end_date)
    ].copy()
    
    data.loc[:, 'MA_20'] = data['Close'].rolling(window=20).mean()
    data.loc[:, 'EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data.loc[:, 'RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data.loc[:, 'MACD'] = ta.trend.MACD(close=data['Close']).macd()
    data.loc[:, 'Volatility'] = data['Close'].pct_change().rolling(20).std()
    
    data.loc[:, 'DayOfWeek'] = data.index.dayofweek
    data.loc[:, 'DayOfMonth'] = data.index.day
    data.loc[:, 'Month'] = data.index.month
    
    data = data.dropna()
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 
                'EMA_20', 'RSI', 'MACD', 'Volatility', 'DayOfWeek', 
                'DayOfMonth', 'Month']
    
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
    
    for i in range(config.time_step, len(scaled_data) - future_steps + 1):
        x_train.append(scaled_data[i - config.time_step:i])
        y_train.append(scaled_data[i:i + future_steps, 3])
    
    return np.array(x_train), np.array(y_train)

def build_and_train_model(scaled_data, config, future_steps):
    x_train, y_train = prepare_multistep_data(scaled_data, config, future_steps)
    
    input_layer = Input(shape=(config.time_step, scaled_data.shape[1]))
    
    x = Bidirectional(LSTM(config.hidden_size, 
                          return_sequences=True,
                          kernel_regularizer=l2(1e-5)))(input_layer)
    x = Dropout(config.dropout_rate)(x)
    
    residual = x
    x = Bidirectional(LSTM(config.hidden_size, 
                          return_sequences=True,
                          kernel_regularizer=l2(1e-5)))(x)
    x = Add()([x, residual]) 
    x = Dropout(config.dropout_rate)(x)
    
    attention = Attention()([x, x])
    attention_flat = Flatten()(attention)
    
    x = Dense(config.hidden_size, activation='relu')(attention_flat)
    x = Dense(config.hidden_size//2, activation='tanh')(x)
    output = Dense(future_steps)(x)
    
    model = Model(inputs=input_layer, outputs=output)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        clipnorm=0.5 
    )
    
    model.compile(optimizer=optimizer,
                 loss='huber')

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            min_delta=1e-5
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7, 
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        x_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epoch,
        validation_split=config.valid_data_rate,
        callbacks=callbacks,
        shuffle=True
    )

    return model, history.history['val_loss'][-1]

def predict_future_prices(model, last_sequence, scaler, future_steps, last_actual_price):
    X = np.array([last_sequence])
    pred_scaled = model.predict(X)
    
    pred_full = np.zeros((future_steps, scaler.scale_.shape[0]))
    pred_full[:, 3] = pred_scaled[0]
    
    predictions = scaler.inverse_transform(pred_full)[:, 3]
    
    pred_std = np.std(predictions) 
    confidence_intervals = np.array([
        predictions - 2 * pred_std,
        predictions + 2 * pred_std
    ]).T
    
    adjustment = last_actual_price - predictions[0]
    adjusted_predictions = predictions + adjustment
    confidence_intervals += adjustment
    
    return adjusted_predictions, confidence_intervals

def plot_predictions(data, predictions, confidence_intervals, days_to_predict):
    plt.style.use('fivethirtyeight')
    
    fig, ax = plt.subplots(figsize=(8, 4))

    last_three_months = data['Close'].last('3M')
    ax.plot(last_three_months.index, last_three_months, 
            label='Historical', color='blue', alpha=0.7)
    
    last_date = data.index[-1]
    prediction_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days_to_predict,
        freq='B'
    )
    
    last_price = data['Close'].iloc[-1]
    price_diff = last_price - predictions[0]
    adjusted_predictions = predictions + price_diff
    adjusted_confidence = confidence_intervals + price_diff
    
    all_dates = [last_date] + list(prediction_dates)
    all_predictions = [last_price] + list(adjusted_predictions)
    
    ax.plot(all_dates, all_predictions,
            label='Prediction', color='red', linestyle='--', linewidth=1)
    
    ax.fill_between(prediction_dates,
                    adjusted_confidence[:, 0],
                    adjusted_confidence[:, 1],
                    color='red', alpha=0.1,
                    label='95% Confidence Interval')
    
    returns = data['Close'].pct_change()
    last_30_returns = returns.last('30D')
    volatility = last_30_returns.std() * np.sqrt(252)
    sharpe = (last_30_returns.mean() * 252) / volatility if volatility != 0 else 0
    
    metrics_text = f'30D Metrics:\nVol: {volatility:.2%}\nSharpe: {sharpe:.2f}'
    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top',
            fontsize=8) 
    
    ax.set_title(f'{data.index[-1].strftime("%Y-%m-%d")} to {prediction_dates[-1].strftime("%Y-%m-%d")}',
                 fontsize=10)
    ax.set_xlabel('Date', fontsize=8)
    ax.set_ylabel('Price ($)', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "temp_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight') 
    plt.close()
    
    return plot_path

def predict_stock(stock_ticker, days_to_predict=60):
    data = load_stock_data(stock_ticker)
    scaled_data, scaler = preprocess_data(data, days_to_predict)
    
    model, val_score = build_and_train_model(scaled_data, config, days_to_predict)
    
    last_sequence = scaled_data[-config.time_step:]
    last_actual_price = data['Close'].values[-1]
    predictions, confidence_intervals = predict_future_prices(
        model, last_sequence, scaler, days_to_predict, last_actual_price
    )
    
    plot_path = plot_predictions(data, predictions, confidence_intervals, days_to_predict)
    
    return plot_path