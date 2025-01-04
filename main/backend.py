import backtrader as bt
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
    """Basic configuration for model training."""
    hidden_size = 128
    lstm_layers = 2
    dropout_rate = 0.2
    time_step = 20
    batch_size = 64
    learning_rate = 0.001
    epoch = 30
    valid_data_rate = 0.15
    random_seed = 42

class EnhancedConfig(Config):
    """Extended configuration including Backtrader parameters."""
    n_splits = 3
    volatility_target = 0.15
    confidence_threshold = 1.5
    max_position = 1.0
    commission = 0.001  # 0.1% commission per trade
    initial_cash = 100000

# Set random seeds
config = EnhancedConfig()
np.random.seed(config.random_seed)
tf.random.set_seed(config.random_seed)

# --------------------------------------------------------------------------------
# Backtrader Strategy
# --------------------------------------------------------------------------------

class MLPredictionStrategy(bt.Strategy):
    params = (
        ('prediction_window', 30),
        ('position_size', 1.0),
    )
    
    def __init__(self):
        self.predictions = None
        self.confidence_intervals = None
        self.order = None
        self.current_position = 0
    
    def set_predictions(self, predictions, confidence_intervals):
        self.predictions = predictions
        self.confidence_intervals = confidence_intervals
    
    def next(self):
        if self.predictions is None:
            return
            
        idx = len(self) - 1
        if idx >= len(self.predictions):
            return
            
        target_position = self.calculate_position(idx)
        
        if self.current_position != target_position:
            if self.order:
                self.cancel(self.order)
                
            size = (target_position - self.current_position) * self.params.position_size
            self.order = self.order_target_percent(target=size)
            self.current_position = target_position
    
    def calculate_position(self, idx):
        pred_return = (self.predictions[idx] / self.data.close[0]) - 1
        confidence_width = (self.confidence_intervals[idx][1] - self.confidence_intervals[idx][0]) / self.data.close[0]
        
        if confidence_width > 0:
            position = pred_return / confidence_width
        else:
            position = 0
            
        return np.clip(position, -1, 1)

# --------------------------------------------------------------------------------
# Data Loading and Preprocessing
# --------------------------------------------------------------------------------

def load_stock_data(stock_ticker):
    """Fetch historical stock data and calculate technical indicators."""
    full_data = yf.Ticker(stock_ticker).history(period='max')
    
    start_date = '2010-01-01'
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    data = full_data.loc[(full_data.index >= start_date) & (full_data.index <= end_date)].copy()
    
    # Calculate technical indicators
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(close=data['Close']).macd()
    data['Volatility'] = data['Close'].pct_change().rolling(20).std()
    
    # Add date-based features
    data['DayOfWeek'] = data.index.dayofweek
    data['DayOfMonth'] = data.index.day
    data['Month'] = data.index.month
    
    data = data.dropna()
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'MA_20', 'EMA_20', 'RSI', 'MACD', 'Volatility',
                'DayOfWeek', 'DayOfMonth', 'Month']
    
    return data[features]

def preprocess_data(data, future_steps):
    """Scale the dataset using MinMaxScaler."""
    dataset = data.values
    train_data = dataset[:-future_steps]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    
    scaled_data = scaler.transform(dataset)
    return scaled_data, scaler

def prepare_multistep_data(scaled_data, config, future_steps):
    """Build x and y arrays for multistep forecasting."""
    x_train = []
    y_train = []
    
    for i in range(config.time_step, len(scaled_data) - future_steps + 1):
        x_train.append(scaled_data[i - config.time_step:i])
        y_train.append(scaled_data[i:i + future_steps, 3])
    
    return np.array(x_train), np.array(y_train)

# --------------------------------------------------------------------------------
# Model Training Helpers
# --------------------------------------------------------------------------------

def time_series_cv_split(data: np.ndarray, config: EnhancedConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create time series cross-validation splits."""
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    splits = []
    
    for train_idx, val_idx in tscv.split(data):
        if len(train_idx) > config.time_step:
            splits.append((data[train_idx], data[val_idx]))
    
    return splits

def evaluate_fold(train_data: np.ndarray, val_data: np.ndarray, config: EnhancedConfig, future_steps: int):
    """Train and evaluate model on a single cross-validation fold."""
    x_train, y_train = prepare_multistep_data(train_data, config, future_steps)
    x_val, y_val = prepare_multistep_data(val_data, config, future_steps)
    
    input_layer = Input(shape=(config.time_step, train_data.shape[1]))
    
    x = Bidirectional(LSTM(config.hidden_size, return_sequences=True, kernel_regularizer=l2(1e-5)))(input_layer)
    x = Dropout(config.dropout_rate)(x)
    
    x = Bidirectional(LSTM(config.hidden_size, return_sequences=True, kernel_regularizer=l2(1e-5)))(x)
    x = Dropout(config.dropout_rate)(x)
    
    attention = Attention()([x, x])
    attention_flat = Flatten()(attention)
    
    x = Dense(config.hidden_size, activation='relu')(attention_flat)
    x = Dense(config.hidden_size // 2, activation='tanh')(x)
    
    output = Dense(future_steps)(x)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=0.5),
                 loss='huber')
    
    history = model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       batch_size=config.batch_size,
                       epochs=config.epoch,
                       callbacks=[
                           EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5)
                       ],
                       verbose=0)
    
    return model, {
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
        'train_predictions': model.predict(x_train),
        'val_predictions': model.predict(x_val)
    }

# --------------------------------------------------------------------------------
# Model Building and Training
# --------------------------------------------------------------------------------

def build_and_train_model(scaled_data, config, future_steps):
    """Build and train the LSTM model."""
    x_train, y_train = prepare_multistep_data(scaled_data, config, future_steps)
    
    input_layer = Input(shape=(config.time_step, scaled_data.shape[1]))
    
    x = Bidirectional(LSTM(config.hidden_size, return_sequences=True, kernel_regularizer=l2(1e-5)))(input_layer)
    x = Dropout(0.2)(x)
    
    x = Bidirectional(LSTM(config.hidden_size // 2, return_sequences=True, kernel_regularizer=l2(1e-5)))(x)
    x = Dropout(0.2)(x)
    
    attention = Attention()([x, x])
    attention_flat = Flatten()(attention)
    
    x = Dense(config.hidden_size, activation='relu')(attention_flat)
    x = Dense(config.hidden_size // 2, activation='tanh')(x)
    
    output = Dense(future_steps)(x)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate, clipnorm=0.5),
                 loss='huber')
    
    history = model.fit(x_train, y_train,
                       batch_size=config.batch_size,
                       epochs=config.epoch,
                       validation_split=config.valid_data_rate,
                       callbacks=[
                           EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5)
                       ],
                       verbose=1)
    
    return model, history.history['val_loss'][-1]

# --------------------------------------------------------------------------------
# Prediction and Backtesting
# --------------------------------------------------------------------------------

def predict_future_prices(model, last_sequence, scaler, future_steps, last_actual_price):
    """Predict future prices with confidence intervals."""
    predictions = np.zeros(future_steps)
    current_sequence = last_sequence.copy()
    
    historical_prices = scaler.inverse_transform(last_sequence)[:, 3]
    historical_returns = np.diff(np.log(historical_prices))
    historical_vol = np.std(historical_returns) * np.sqrt(252)
    historical_mean_return = np.mean(historical_returns)
    
    daily_vol = historical_vol / np.sqrt(252)
    mean_reversion_strength = 0.1
    
    cumulative_return = 0
    
    for i in range(future_steps):
        model_input = np.array([current_sequence])
        base_pred = model.predict(model_input, verbose=0)[0][0]
        
        random_component = np.random.normal(0, daily_vol)
        mean_reversion = mean_reversion_strength * (historical_mean_return - cumulative_return)
        momentum_factor = 0.05 * (predictions[i-1] - predictions[i-2]) if i > 1 else 0
        
        daily_return = random_component + mean_reversion + momentum_factor
        cumulative_return += daily_return
        
        current_pred = base_pred * (1 + daily_return)
        predictions[i] = current_pred
        
        new_row = current_sequence[-1].copy()
        new_row[3] = current_pred
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = new_row
    
    pred_full = np.zeros((future_steps, scaler.scale_.shape[0]))
    pred_full[:, 3] = predictions
    price_predictions = scaler.inverse_transform(pred_full)[:, 3]
    
    time_scalar = np.sqrt(np.arange(1, future_steps + 1))
    vol_adjusted = historical_vol / np.sqrt(252)
    pred_std = last_actual_price * vol_adjusted * time_scalar
    confidence_intervals = np.array([
        price_predictions - 1.96 * pred_std,
        price_predictions + 1.96 * pred_std
    ]).T
    
    adjustment = last_actual_price - price_predictions[0]
    adjusted_predictions = price_predictions + adjustment
    confidence_intervals += adjustment
    
    confidence_intervals = np.maximum(confidence_intervals, 0)
    adjusted_predictions = np.maximum(adjusted_predictions, 0)
    
    return adjusted_predictions, confidence_intervals

def run_backtest(data, predictions, confidence_intervals, config):
    """Run backtest using Backtrader."""
    cerebro = bt.Cerebro()
    
    # Add data feed
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    
    # Add strategy
    strategy = MLPredictionStrategy
    cerebro.addstrategy(strategy, prediction_window=len(predictions))
    
    # Set initial cash and commission
    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(commission=config.commission)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    
    # Run backtest
    results = cerebro.run()
    strategy = results[0]
    strategy.set_predictions(predictions, confidence_intervals)
    
    backtest_results = pd.DataFrame({
        'Date': pd.date_range(start=data.index[-1], periods=len(predictions), freq='B'),
        'Prediction': predictions,
        'Lower_CI': confidence_intervals[:, 0],
        'Upper_CI': confidence_intervals[:, 1],
    })
    
    return backtest_results

def plot_predictions(data, predictions, days_to_predict):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot historical data (last 60 days)
    last_three_months = data['Close'].tail(60)
    ax.plot(last_three_months.index, last_three_months, 
            label='Historical', color='blue', alpha=0.7)
    
    # Set up dates for predictions
    last_date = data.index[-1]
    prediction_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days_to_predict,
        freq='B'
    )
    
    # Combine historical and prediction data for plotting
    last_price = data['Close'].iloc[-1]
    all_dates = [last_date] + list(prediction_dates)
    all_predictions = [last_price] + list(predictions)
    
    # Plot predictions
    ax.plot(all_dates, all_predictions, 
            label='Prediction', color='red', 
            linestyle='--', linewidth=1)
    
    # Calculate and display metrics
    returns = data['Close'].pct_change()
    last_30_returns = returns.tail(30)
    volatility = last_30_returns.std() * np.sqrt(252)
    sharpe = (last_30_returns.mean() * 252) / volatility if volatility != 0 else 0
    
    metrics_text = f'30D Metrics:\nVol: {volatility:.2%}\nSharpe: {sharpe:.2f}'
    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top',
            fontsize=8)
    
    # Customize plot appearance
    ax.set_title(f'{data.index[-1].strftime("%Y-%m-%d")} to {prediction_dates[-1].strftime("%Y-%m-%d")}',
                fontsize=10)
    ax.set_xlabel('Date', fontsize=8)
    ax.set_ylabel('Price ($)', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Save and close plot
    plt.tight_layout()
    plot_path = "temp_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path
                                   
def calculate_position_sizes(predictions: np.ndarray, confidence_intervals: np.ndarray,
                           config: EnhancedConfig, current_volatility: float) -> np.ndarray:
    """Calculate position sizes based on predictions and confidence intervals."""
    mid_price = (confidence_intervals[:, 1] + confidence_intervals[:, 0]) / 2
    interval_width = confidence_intervals[:, 1] - confidence_intervals[:, 0]
    
    signal_strength = (predictions - mid_price) / (interval_width / 2)
    signal_strength = np.clip(signal_strength, -1, 1)
    
    volatility_scalar = min(config.volatility_target / current_volatility, 2.0)
    raw_positions = signal_strength * volatility_scalar
    
    positions = np.clip(raw_positions, -config.max_position, config.max_position)
    smoothed_positions = pd.Series(positions).ewm(span=5).mean().values
    
    return smoothed_positions

def backtest_strategy(data: pd.DataFrame, predictions: np.ndarray,
                     confidence_intervals: np.ndarray, config: EnhancedConfig) -> pd.DataFrame:
    """Backtest the trading strategy using Backtrader and position sizing."""
    returns = data['Close'].pct_change()
    realized_vol = returns.rolling(30).std() * np.sqrt(252)
    
    positions = calculate_position_sizes(
        predictions,
        confidence_intervals,
        config,
        realized_vol.iloc[-1]
    )
    
    backtest_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=len(predictions),
        freq='B'
    )
    
    results = pd.DataFrame({
        'Date': backtest_dates,
        'Prediction': predictions,
        'Position': positions,
        'Lower_CI': confidence_intervals[:, 0],
        'Upper_CI': confidence_intervals[:, 1]
    })
    
    results['Predicted_Return'] = np.log(results['Prediction'] / results['Prediction'].shift(1))
    results['Strategy_Return'] = results['Position'].shift(1) * results['Predicted_Return']
    results['Cumulative_Return'] = np.exp(results['Strategy_Return'].cumsum()) - 1
    results['Rolling_Volatility'] = results['Strategy_Return'].rolling(30).std() * np.sqrt(252)
    results['Rolling_Sharpe'] = (results['Strategy_Return'].rolling(30).mean() * 252 /
                                (results['Rolling_Volatility'] + 1e-6))
    
    return results

def predict_stock(stock_ticker, days_to_predict=60):
    """Main function to load data, train model, and generate predictions with backtesting."""
    data = load_stock_data(stock_ticker)
    scaled_data, scaler = preprocess_data(data, days_to_predict)
    
    model, _ = build_and_train_model(scaled_data, config, days_to_predict)
    
    last_sequence = scaled_data[-config.time_step:]
    last_actual_price = data['Close'].values[-1]
    
    predictions, confidence_intervals = predict_future_prices(
        model, last_sequence, scaler, days_to_predict, last_actual_price
    )
    
    backtest_results = backtest_strategy(data, predictions, confidence_intervals, config)
    bt_results = run_backtest(data, predictions, confidence_intervals, config)
    
    combined_results = pd.merge(
        backtest_results,
        bt_results,
        on='Date',
        suffixes=('', '_bt')
    )

    print(combined_results)
    
    plot_path = plot_predictions(data, predictions, days_to_predict)
    
    return plot_path, combined_results