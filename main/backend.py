import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Attention, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from typing import Tuple, List
from sklearn.model_selection import TimeSeriesSplit
import ta

# --------------------------------------------------------------------------------
# Configuration classes
# --------------------------------------------------------------------------------

class Config:
    """
    Basic configuration for model training.
    """
    hidden_size = 64         # Number of units in LSTM layers
    lstm_layers = 2          # Number of LSTM layers
    dropout_rate = 0.2       # Dropout rate for regularization
    time_step = 30           # Number of past days used for each training input
    batch_size = 32
    learning_rate = 0.001
    epoch = 30
    valid_data_rate = 0.2
    random_seed = 42


class EnhancedConfig(Config):
    """
    Extended configuration for any additional features or parameters.
    """
    n_splits = 3
    volatility_target = 0.15
    confidence_threshold = 1.5
    max_position = 1.0


# Instantiate config and set seeds
config = EnhancedConfig()
np.random.seed(config.random_seed)
tf.random.set_seed(config.random_seed)

# --------------------------------------------------------------------------------
# Data Loading and Preprocessing
# --------------------------------------------------------------------------------

def load_stock_data(stock_ticker):
    """
    Fetch historical stock data from Yahoo Finance,
    add technical indicators, and return a feature DataFrame.
    """
    full_data = yf.Ticker(stock_ticker).history(period='max')
    
    # Filter data between 2010-01-01 and today
    start_date = '2010-01-01'
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    data = full_data.loc[(full_data.index >= start_date) & (full_data.index <= end_date)].copy()
    
    # Calculate basic technical indicators
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(close=data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(20).std()
    
    # Add date-based features
    data['DayOfWeek'] = data.index.dayofweek
    data['DayOfMonth'] = data.index.day
    data['Month'] = data.index.month
    
    # Drop rows with NaNs (due to rolling calculations)
    data = data.dropna()
    
    # Select features
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA_20', 'EMA_20', 'RSI', 'MACD', 'Volatility',
        'DayOfWeek', 'DayOfMonth', 'Month'
    ]
    
    return data[features]


def preprocess_data(data, future_steps):
    """
    Scale the dataset using MinMaxScaler. Leave out the last 'future_steps' rows
    from the fit to simulate a real future prediction scenario.
    """
    dataset = data.values
    train_data = dataset[:-future_steps]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    
    scaled_data = scaler.transform(dataset)
    return scaled_data, scaler


def prepare_multistep_data(scaled_data, config, future_steps):
    """
    Build x (past data) and y (future 'Close' values) arrays for multistep forecasting.
    """
    x_train = []
    y_train = []
    
    for i in range(config.time_step, len(scaled_data) - future_steps + 1):
        x_train.append(scaled_data[i - config.time_step:i])
        y_train.append(scaled_data[i:i + future_steps, 3])  # 'Close' column index = 3
    
    return np.array(x_train), np.array(y_train)


# --------------------------------------------------------------------------------
# Model Training Helpers
# --------------------------------------------------------------------------------

def time_series_cv_split(data: np.ndarray, config: EnhancedConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time series cross-validation splits to maintain temporal order.
    """
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    splits = []
    
    for train_idx, val_idx in tscv.split(data):
        # Ensure we have enough historical data for the time step
        if len(train_idx) > config.time_step:
            splits.append((data[train_idx], data[val_idx]))
    
    return splits


def evaluate_fold(train_data: np.ndarray, val_data: np.ndarray, config: EnhancedConfig, future_steps: int):
    """
    Train and evaluate model on a single cross-validation fold.
    """
    # Prepare data
    x_train, y_train = prepare_multistep_data(train_data, config, future_steps)
    x_val, y_val = prepare_multistep_data(val_data, config, future_steps)
    
    # Model architecture
    input_layer = Input(shape=(config.time_step, train_data.shape[1]))
    
    x = Bidirectional(
        LSTM(config.hidden_size, return_sequences=True, kernel_regularizer=l2(1e-5))
    )(input_layer)
    x = Dropout(config.dropout_rate)(x)
    
    x = Bidirectional(
        LSTM(config.hidden_size, return_sequences=True, kernel_regularizer=l2(1e-5))
    )(x)
    x = Dropout(config.dropout_rate)(x)
    
    attention = Attention()([x, x])
    attention_flat = Flatten()(attention)
    
    x = Dense(config.hidden_size, activation='relu')(attention_flat)
    x = Dense(config.hidden_size // 2, activation='tanh')(x)
    
    output = Dense(future_steps)(x)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate, clipnorm=0.5
    )
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=optimizer, loss='huber')
    
    # Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=config.batch_size,
        epochs=config.epoch,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=7)
        ],
        verbose=0
    )
    
    metrics = {
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
        'train_predictions': model.predict(x_train),
        'val_predictions': model.predict(x_val)
    }
    
    return model, metrics


# --------------------------------------------------------------------------------
# Model Building and Training
# --------------------------------------------------------------------------------

def build_and_train_model(scaled_data, config, future_steps):
    """
    Build the LSTM model, then train with early stopping and learning rate reduction.
    """
    x_train, y_train = prepare_multistep_data(scaled_data, config, future_steps)
    
    input_layer = Input(shape=(config.time_step, scaled_data.shape[1]))
    
    x = Bidirectional(
        LSTM(config.hidden_size, return_sequences=True, kernel_regularizer=l2(1e-5))
    )(input_layer)
    x = Dropout(0.2)(x)
    
    x = Bidirectional(
        LSTM(config.hidden_size // 2, return_sequences=True, kernel_regularizer=l2(1e-5))
    )(x)
    x = Dropout(0.2)(x)
    
    attention = Attention()([x, x])
    attention_flat = Flatten()(attention)
    
    x = Dense(config.hidden_size, activation='relu')(attention_flat)
    x = Dense(config.hidden_size // 2, activation='tanh')(x)
    
    output = Dense(future_steps)(x)
    
    model = Model(inputs=input_layer, outputs=output)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate, clipnorm=0.5
    )
    
    model.compile(optimizer=optimizer, loss='huber')
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True, min_delta=1e-5
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.7, patience=7, min_lr=1e-6, verbose=1
        )
    ]
    
    history = model.fit(
        x_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epoch,
        validation_split=config.valid_data_rate,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    
    return model, history.history['val_loss'][-1]


# --------------------------------------------------------------------------------
# Prediction and Plotting
# --------------------------------------------------------------------------------

def predict_future_prices(model, last_sequence, scaler, future_steps, last_actual_price):
    """
    Predict future 'Close' prices and apply a simple offset to avoid extreme jumps.
    """
    single_prediction = model.predict(np.array([last_sequence]), verbose=0)[0]
    
    pred_full = np.zeros((future_steps, scaler.scale_.shape[0]))
    pred_full[:, 3] = single_prediction
    
    one_shot_price_pred = scaler.inverse_transform(pred_full)[:, 3]
    
    price_diff = last_actual_price - one_shot_price_pred[0]
    adjusted_pred = one_shot_price_pred + price_diff
    adjusted_pred = np.maximum(adjusted_pred, 0)
    
    return adjusted_pred


def plot_predictions(data, predictions, days_to_predict):
    """
    Plot the last 3 months of historical data plus the future predictions.
    Displays basic volatility and Sharpe ratio metrics for context.
    """
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(8, 4))
    
    last_three_months = data['Close'].last('3M')
    ax.plot(last_three_months.index, last_three_months, label='Historical', color='blue', alpha=0.7)
    
    last_date = data.index[-1]
    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=days_to_predict,
                                     freq='B')
    
    last_price = data['Close'].iloc[-1]
    
    all_dates = [last_date] + list(prediction_dates)
    all_predictions = [last_price] + list(predictions)
    
    ax.plot(all_dates, all_predictions, label='Prediction', color='red', linestyle='--', linewidth=1)
    
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


# --------------------------------------------------------------------------------
# Main Prediction Function
# --------------------------------------------------------------------------------

def predict_stock(stock_ticker, days_to_predict=60):
    """
    Main function to:
      1) Load stock data
      2) Preprocess and train model
      3) Predict future prices
      4) Generate plot
    """
    data = load_stock_data(stock_ticker)
    scaled_data, scaler = preprocess_data(data, days_to_predict)
    
    model, _ = build_and_train_model(scaled_data, config, days_to_predict)
    
    last_sequence = scaled_data[-config.time_step:]
    last_actual_price = data['Close'].values[-1]
    
    predictions = predict_future_prices(model, last_sequence, scaler, days_to_predict, last_actual_price)
    
    plot_path = plot_predictions(data, predictions, days_to_predict)
    
    return plot_path