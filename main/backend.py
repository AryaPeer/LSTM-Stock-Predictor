import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
import pandas_datareader.data as web
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Tuple
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Attention,
    Dropout,
    Dense,
    Flatten,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configuration for model hyperparameters and training settings
class Config:
    # Input/output dimensions
    time_step: int = 60          # Length of input sequence window
    future_steps: int = 10       # Number of days to forecast ahead
    
    # Model architecture parameters
    cnn_filters: int = 32        # Number of CNN filters
    kernel_size: int = 3         # Size of CNN kernel
    lstm_units: int = 128        # Units in first LSTM layer
    dropout_rate: float = 0.2    # Dropout for regularization
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100            # Maximum epochs (early stopping usually triggers before this)
    valid_split: float = 0.15    # Validation data percentage
    random_seed: int = 42        # For reproducibility

# Extended configuration with cross-validation settings
class CVConfig(Config):
    # Time series cross-validation settings
    n_splits: int = 5            # Number of CV folds

# Set random seeds for reproducible results
np.random.seed(Config.random_seed)
tf.random.set_seed(Config.random_seed)

# Encodes cyclical features like day of week or month of year
def _cyclical_encode(series: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    # Transform cyclical data to sine/cosine components
    # Helps capture periodicity in time-based features
    radians = 2 * np.pi * series / period
    return np.sin(radians), np.cos(radians)

# Load and prepare stock data with technical indicators
def load_stock_data(ticker: str, start: str = "2010-01-01") -> pd.DataFrame:
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Load primary stock data
    df = web.DataReader(ticker, "stooq", start, end)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} from Stooq")
    df = df.sort_index()  # Stooq returns descending order
    df.index = pd.to_datetime(df.index)

    # Add market context (S&P 500)
    sp500 = web.DataReader("^spx", "stooq", start, end)["Close"].sort_index()
    sp500.name = "SP500_Close"
    df = df.join(sp500, how="left")

    # Calculate technical indicators

    # Moving averages
    df["MA_20"]   = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA_20"]  = ta.trend.ema_indicator(df["Close"], window=20)
    df["MA_50"]   = ta.trend.sma_indicator(df["Close"], window=50)
    df["MA_100"]  = ta.trend.sma_indicator(df["Close"], window=100)

    # Relative Strength Index
    df["RSI_14"]  = ta.momentum.rsi(df["Close"], window=14)
    # Moving Average Convergence Divergence
    df["MACD"]    = ta.trend.macd(df["Close"])

    # Bollinger Bands
    bb            = ta.volatility.BollingerBands(df["Close"], window=20)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"]  = bb.bollinger_lband()

    # Stochastic Oscillator
    df["Stoch"]   = ta.momentum.stoch(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
    # Average True Range
    df["ATR"]     = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    # On Balance Volume
    df["OBV"]     = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    # Money Flow Index
    df["MFI"]     = ta.volume.money_flow_index(df["High"], df["Low"], df["Close"], df["Volume"], window=14)

    # Calculate returns and volatility
    df["Return"] = df["Close"].pct_change()
    df["Volatility_20"] = df["Return"].rolling(20).std()

    # Add calendar features with cyclical encoding
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["DayOfWeek_sin"], df["DayOfWeek_cos"] = _cyclical_encode(df["DayOfWeek"], 7)
    df["Month_sin"],    df["Month_cos"] = _cyclical_encode(df["Month"], 12)

    # Clean data and select relevant features
    df = df.dropna()
    feature_cols = [
        "Open", 
        "High", 
        "Low", 
        "Close", 
        "Volume",
        "MA_20", 
        "EMA_20", 
        "MA_50", 
        "MA_100", 
        "RSI_14", 
        "MACD",
        "BB_High", 
        "BB_Low", 
        "Stoch", 
        "ATR", 
        "OBV", 
        "MFI",
        "Return", 
        "Volatility_20", 
        "SP500_Close",
        "DayOfWeek_sin", 
        "DayOfWeek_cos", 
        "Month_sin", 
        "Month_cos",
    ]
    return df[feature_cols]

# Scale data for neural network processing
def preprocess_data(data: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, MinMaxScaler]:
    # Normalize all features to 0-1 range for better model training
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values)
    return scaled, scaler

# Create time series sequences for model training
def _make_sequences(scaled: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    # Generate input windows and corresponding target values
    X, y = [], []
    for i in range(cfg.time_step, len(scaled) - cfg.future_steps + 1):
        X.append(scaled[i - cfg.time_step : i])
        y.append(scaled[i : i + cfg.future_steps, 3])  # Close column index 3
    return np.asarray(X), np.asarray(y)

# Set up time series cross-validation
def time_series_cv(data: np.ndarray, cv_cfg: CVConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Create time-aware CV splits maintaining temporal order
    splitter = TimeSeriesSplit(n_splits=cv_cfg.n_splits)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in splitter.split(data):
        if len(train_idx) < cv_cfg.time_step:
            continue
        folds.append((data[train_idx], data[val_idx]))
    return folds

# Create the hybrid CNN-LSTM model architecture
def _build_model(n_features: int, cfg: Config) -> Model:
    # Build a hybrid model combining CNN, BiLSTM and attention mechanisms
    inp = Input(shape=(cfg.time_step, n_features))
    
    # CNN layer to capture local patterns
    x = Conv1D(cfg.cnn_filters, cfg.kernel_size, activation="relu", padding="same")(inp)
    x = MaxPooling1D(2)(x)
    
    # Bidirectional LSTM layers to capture temporal dependencies
    x = Bidirectional(LSTM(cfg.lstm_units, return_sequences=True, kernel_regularizer=l2(1e-5)))(x)
    x = Dropout(cfg.dropout_rate)(x)
    x = Bidirectional(LSTM(cfg.lstm_units // 2, return_sequences=True, kernel_regularizer=l2(1e-5)))(x)
    x = Dropout(cfg.dropout_rate)(x)
    
    # Attention layer to focus on important time steps
    context = Attention()([x, x])
    context = Flatten()(context)
    
    # Dense layers for final prediction
    x = Dense(128, activation="relu")(context)
    x = Dense(64, activation="tanh")(x)
    out = Dense(cfg.future_steps)(x)
    
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate, clipnorm=0.5), loss="huber")
    return model

# Train the model with early stopping
def train_model(scaled: np.ndarray, cfg: Config) -> Tuple[Model, float]:
    # Prepare sequences and train model with validation
    X, y = _make_sequences(scaled, cfg)
    model = _build_model(scaled.shape[1], cfg)
    
    hist = model.fit(
        X,
        y,
        validation_split=cfg.valid_split,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        callbacks=[
            #Reduce learning rate on plateau + Early stopping
            EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_loss"),
        ],
        verbose=2,
    )
    return model, hist.history["val_loss"][-1]

# Generate future price predictions
def predict_future(model: Model, last_sequence: np.ndarray, scaler: MinMaxScaler, cfg: Config) -> np.ndarray:
    # Predict future prices with realistic market adjustments
    seq = last_sequence.copy()
    preds_scaled = np.zeros(cfg.future_steps)

    # Get baseline information for adjustments
    last_actual_price = scaler.inverse_transform(seq[-1].reshape(1, -1))[0, 3]
    historical_prices = scaler.inverse_transform(seq)[:, 3]
    historical_returns = np.diff(np.log(historical_prices))
    historical_vol = np.std(historical_returns) * np.sqrt(252) if historical_returns.size else 0.0
    historical_mean = np.mean(historical_returns) if historical_returns.size else 0.0
    daily_vol = historical_vol / np.sqrt(252) if historical_vol else 0.0
    mean_rev_strength = 0.1
    cum_return = 0.0

    # Generate predictions with adjustments for realism
    for i in range(cfg.future_steps):
        # Get base prediction from model
        base_scaled = model.predict(np.expand_dims(seq, 0), verbose=0)[0][0]
        
        # Add realistic market effects: random movement, mean reversion, momentum
        rand = np.random.normal(0, daily_vol)
        mean_rev = mean_rev_strength * (historical_mean - cum_return)
        momentum = 0.05 * (preds_scaled[i-1] - preds_scaled[i-2]) if i > 1 else 0.0
        
        # Calculate daily return using random movement, mean reversion, and momentum
        daily_ret = rand + mean_rev + momentum
        cum_return += daily_ret
        
        # Apply daily return to base prediction
        adj_scaled = base_scaled * (1 + daily_ret)
        preds_scaled[i] = adj_scaled
        
        # Update sequence for next prediction
        new_row = seq[-1].copy()
        new_row[3] = adj_scaled
        seq = np.roll(seq, -1, axis=0)
        seq[-1] = new_row

    # Convert scaled predictions back to original price range
    template = np.zeros((cfg.future_steps, scaler.scale_.shape[0]))
    template[:, 3] = preds_scaled
    preds = scaler.inverse_transform(template)[:, 3]
    preds += (last_actual_price - preds[0])
    preds[preds < 0] = 0.0
    return preds

# Complete forecasting pipeline
def forecast_stock(ticker: str, cfg: Config = Config()) -> Tuple[pd.DataFrame, np.ndarray]:
    # End-to-end pipeline from data loading to prediction
    data = load_stock_data(ticker)
    scaled, scaler = preprocess_data(data, cfg)
    model, _ = train_model(scaled, cfg)
    last_seq = scaled[-cfg.time_step :]
    preds = predict_future(model, last_seq, scaler, cfg)
    return data, preds

# Create visualization of forecast
def plot_forecast(data: pd.DataFrame, preds: np.ndarray, cfg: Config, save_path: str = "forecast.png") -> str:
    # Generate plot showing historical data and forecast
    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot historical data
    hist = data["Close"].tail(60)
    ax.plot(hist.index, hist, label="Historical", color="blue", alpha=0.7)
    
    # Plot forecast
    last_date = data.index[-1]
    pred_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=cfg.future_steps, freq="B")
    ax.plot([last_date] + list(pred_dates), [hist.iloc[-1]] + preds.tolist(), 
            label="Forecast", color="red", linestyle="--", linewidth=1)
    
    # Add labels and styling
    ax.set_title(f"{cfg.future_steps}-Day Price Forecast", fontsize=10)
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Price", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
