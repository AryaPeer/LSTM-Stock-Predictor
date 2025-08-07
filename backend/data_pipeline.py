# Data Pipeline

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import ta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import logging
import pickle
import hashlib
from pathlib import Path
from scipy import stats  # noqa: F401
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    start_date: str = "2010-01-01"
    end_date: str = None
    cache_dir: str = "cache"
    cache_ttl: int = 3600  # seconds

    # Feature engineering
    use_technical_indicators: bool = True
    use_pattern_recognition: bool = True
    use_market_indicators: bool = True
    use_fourier_features: bool = True
    use_statistical_features: bool = True

    # Technical indicators periods
    sma_periods: List[int] = None
    ema_periods: List[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: int = 2

    def __post_init__(self):
        if self.end_date is None:
            self.end_date = datetime.now().strftime("%Y-%m-%d")
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 50, 100, 200]
        if self.ema_periods is None:
            self.ema_periods = [12, 26, 50, 100]


class PatternRecognizer:

    @staticmethod
    def detect_support_resistance(prices: pd.Series, window: int = 20) -> Dict[str, List[float]]:
        # Find local maxima and minima
        peaks, _ = find_peaks(prices.values, distance=window)
        troughs, _ = find_peaks(-prices.values, distance=window)

        resistance_levels = prices.iloc[peaks].values if len(peaks) > 0 else []
        support_levels = prices.iloc[troughs].values if len(troughs) > 0 else []

        # Cluster nearby levels
        resistance_levels = PatternRecognizer._cluster_levels(resistance_levels)
        support_levels = PatternRecognizer._cluster_levels(support_levels)

        return {
            'resistance': resistance_levels[:5],  # Top 5 levels
            'support': support_levels[:5]
        }

    @staticmethod
    def _cluster_levels(levels: np.ndarray, threshold: float = 0.02) -> List[float]:
        if len(levels) == 0:
            return []

        levels = np.sort(levels)
        clustered = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]

        if current_cluster:
            clustered.append(np.mean(current_cluster))

        return sorted(clustered, reverse=True)

    @staticmethod
    def detect_chart_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        patterns = {}
        close = df['Close'].values

        # Head and Shoulders
        patterns['head_shoulders'] = PatternRecognizer._detect_head_shoulders(close)

        # Double Top/Bottom
        patterns['double_top'] = PatternRecognizer._detect_double_pattern(close, 'top')
        patterns['double_bottom'] = PatternRecognizer._detect_double_pattern(close, 'bottom')

        # Triangle patterns
        patterns['ascending_triangle'] = PatternRecognizer._detect_triangle(df, 'ascending')
        patterns['descending_triangle'] = PatternRecognizer._detect_triangle(df, 'descending')

        # Flag and Pennant
        patterns['flag'] = PatternRecognizer._detect_flag(df)

        return patterns

    @staticmethod
    def _detect_head_shoulders(prices: np.ndarray, window: int = 20) -> Dict[str, Any]:
        if len(prices) < window * 3:
            return {'detected': False}

        # Find peaks
        peaks, properties = find_peaks(prices, distance=window, prominence=prices.std() * 0.5)

        if len(peaks) >= 3:
            # Check for head and shoulders formation
            for i in range(len(peaks) - 2):
                left_shoulder = prices[peaks[i]]
                head = prices[peaks[i + 1]]
                right_shoulder = prices[peaks[i + 2]]

                # Head should be higher than shoulders
                if head > left_shoulder and head > right_shoulder:
                    # Shoulders should be roughly equal
                    if abs(left_shoulder - right_shoulder) / left_shoulder < 0.05:
                        return {
                            'detected': True,
                            'type': 'bearish',
                            'confidence': 0.7,
                            'neckline': min(prices[peaks[i]:peaks[i + 2]])
                        }

        return {'detected': False}

    @staticmethod
    def _detect_double_pattern(prices: np.ndarray, pattern_type: str = 'top') -> Dict[str, Any]:
        window = 20

        if pattern_type == 'top':
            peaks, _ = find_peaks(prices, distance=window)
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    if abs(prices[peaks[i]] - prices[peaks[i + 1]]) / prices[peaks[i]] < 0.03:
                        return {
                            'detected': True,
                            'type': 'bearish',
                            'confidence': 0.6,
                            'level': prices[peaks[i]]
                        }
        else:
            troughs, _ = find_peaks(-prices, distance=window)
            if len(troughs) >= 2:
                for i in range(len(troughs) - 1):
                    if abs(prices[troughs[i]] - prices[troughs[i + 1]]) / prices[troughs[i]] < 0.03:
                        return {
                            'detected': True,
                            'type': 'bullish',
                            'confidence': 0.6,
                            'level': prices[troughs[i]]
                        }

        return {'detected': False}

    @staticmethod
    def _detect_triangle(df: pd.DataFrame, triangle_type: str) -> Dict[str, Any]:
        high = df['High'].values
        low = df['Low'].values

        if len(df) < 30:
            return {'detected': False}

        # Calculate trendlines
        x = np.arange(len(high))
        high_slope, high_intercept = np.polyfit(x, high, 1)
        low_slope, low_intercept = np.polyfit(x, low, 1)

        if triangle_type == 'ascending':
            # Flat resistance, rising support
            if abs(high_slope) < 0.001 and low_slope > 0.001:
                return {
                    'detected': True,
                    'type': 'bullish',
                    'confidence': 0.5,
                    'breakout_level': high[-1]
                }
        elif triangle_type == 'descending':
            # Falling resistance, flat support
            if high_slope < -0.001 and abs(low_slope) < 0.001:
                return {
                    'detected': True,
                    'type': 'bearish',
                    'confidence': 0.5,
                    'breakdown_level': low[-1]
                }

        return {'detected': False}

    @staticmethod
    def _detect_flag(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
        if len(df) < lookback:
            return {'detected': False}

        recent = df.tail(lookback)
        prior = df.iloc[-lookback*2:-lookback]

        # Check for strong move followed by consolidation
        prior_return = (prior['Close'].iloc[-1] - prior['Close'].iloc[0]) / prior['Close'].iloc[0]
        recent_volatility = recent['Close'].pct_change().std()
        prior_volatility = prior['Close'].pct_change().std()

        if abs(prior_return) > 0.1 and recent_volatility < prior_volatility * 0.5:
            return {
                'detected': True,
                'type': 'continuation',
                'direction': 'bullish' if prior_return > 0 else 'bearish',
                'confidence': 0.6
            }

        return {'detected': False}


class FeatureEngineer:

    @staticmethod
    def add_fourier_features(df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        close_fft = np.fft.fft(df['Close'].values)
        frequencies = np.fft.fftfreq(len(df))

        # Get top frequency components
        power = np.abs(close_fft) ** 2
        top_freq_idx = np.argsort(power)[-n_components:]

        for i, idx in enumerate(top_freq_idx):
            df[f'fourier_freq_{i}'] = frequencies[idx]
            df[f'fourier_power_{i}'] = power[idx] / power.sum()

        return df

    @staticmethod
    def add_statistical_features(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        if windows is None:
            windows = [5, 10, 20, 50]

        for window in windows:
            # Skewness and kurtosis
            df[f'skew_{window}'] = df['Return'].rolling(window).skew()
            df[f'kurt_{window}'] = df['Return'].rolling(window).kurt()

            # Z-score
            rolling_mean = df['Close'].rolling(window).mean()
            rolling_std = df['Close'].rolling(window).std()
            df[f'zscore_{window}'] = (df['Close'] - rolling_mean) / rolling_std

            # Efficiency ratio
            direction = abs(df['Close'] - df['Close'].shift(window))
            volatility = df['Close'].diff().abs().rolling(window).sum()
            df[f'efficiency_{window}'] = direction / volatility

        return df

    @staticmethod
    def add_market_microstructure(df: pd.DataFrame) -> pd.DataFrame:
        # Spread
        df['spread'] = df['High'] - df['Low']
        df['spread_pct'] = df['spread'] / df['Close']

        # Volume-related features
        df['volume_ma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        df['dollar_volume'] = df['Close'] * df['Volume']

        # Price efficiency
        df['close_to_high'] = (df['High'] - df['Close']) / df['High']
        df['close_to_low'] = (df['Close'] - df['Low']) / df['Low']

        # Intraday momentum
        df['intraday_momentum'] = (df['Close'] - df['Open']) / df['Open']

        return df

    @staticmethod
    def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        # Volatility regime
        df['volatility_regime'] = pd.qcut(df['Volatility_20'], q=3, labels=['low', 'medium', 'high'])

        # Trend regime using multiple timeframes
        df['trend_short'] = np.where(df['MA_20'] > df['MA_50'], 1, -1)
        df['trend_long'] = np.where(df['MA_50'] > df['MA_100'], 1, -1)

        # Volume regime
        df['volume_regime'] = np.where(df['volume_ratio'] > 1.5, 'high',
                                       np.where(df['volume_ratio'] < 0.5, 'low', 'normal'))

        return df


class DataPipeline:

    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.pattern_recognizer = PatternRecognizer()
        self.feature_engineer = FeatureEngineer()

    def get_cache_key(self, ticker: str, **kwargs) -> str:
        key_data = f"{ticker}_{self.config.start_date}_{self.config.end_date}_{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()

    @lru_cache(maxsize=32)
    def load_data(self, ticker: str, use_cache: bool = True) -> pd.DataFrame:
        cache_key = self.get_cache_key(ticker)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # Check cache
        if use_cache and cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.seconds < self.config.cache_ttl:
                logger.info(f"Loading {ticker} from cache")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # Load fresh data
        logger.info(f"Loading {ticker} from data source")
        df = self._load_raw_data(ticker)

        # Process data
        df = self._add_base_features(df)

        if self.config.use_technical_indicators:
            df = self._add_technical_indicators(df)

        if self.config.use_pattern_recognition:
            df = self._add_pattern_features(df)

        if self.config.use_market_indicators:
            df = self._add_market_features(df, ticker)

        if self.config.use_fourier_features:
            df = self.feature_engineer.add_fourier_features(df)

        if self.config.use_statistical_features:
            df = self.feature_engineer.add_statistical_features(df)

        # Add market microstructure
        df = self.feature_engineer.add_market_microstructure(df)

        # Add regime features
        df = self.feature_engineer.add_regime_features(df)

        # Clean data
        df = df.dropna()

        # Cache processed data
        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)

        return df

    def _load_raw_data(self, ticker: str) -> pd.DataFrame:
        df = web.DataReader(ticker, 'stooq', self.config.start_date, self.config.end_date)

        if df.empty:
            raise ValueError(f"No data available for {ticker}")

        df = df.sort_index()
        df.index = pd.to_datetime(df.index)

        return df

    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Returns
        df['Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Volatility
        df['Volatility_5'] = df['Return'].rolling(5).std()
        df['Volatility_20'] = df['Return'].rolling(20).std()
        df['Volatility_60'] = df['Return'].rolling(60).std()

        # Calendar features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['DayOfMonth'] = df.index.day
        df['WeekOfYear'] = df.index.isocalendar().week

        # Cyclical encoding
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Moving averages
        for period in self.config.sma_periods:
            df[f'MA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)

        for period in self.config.ema_periods:
            df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)

        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=self.config.rsi_period)

        # MACD
        macd = ta.trend.MACD(df['Close'],
                             window_fast=self.config.macd_fast,
                             window_slow=self.config.macd_slow,
                             window_sign=self.config.macd_signal)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'],
                                          window=self.config.bb_period,
                                          window_dev=self.config.bb_std)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = bb.bollinger_wband()
        df['BB_pct'] = bb.bollinger_pband()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_k'] = stoch.stoch()
        df['Stoch_d'] = stoch.stoch_signal()

        # ATR
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # ADX
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

        # OBV
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

        # MFI
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])

        # Williams %R
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])

        # CCI
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Support and resistance levels
        levels = self.pattern_recognizer.detect_support_resistance(df['Close'])

        # Distance to nearest support/resistance
        if levels['resistance']:
            df['dist_to_resistance'] = df['Close'].apply(
                lambda x: min([abs(x - r) / x for r in levels['resistance']])
            )

        if levels['support']:
            df['dist_to_support'] = df['Close'].apply(
                lambda x: min([abs(x - s) / x for s in levels['support']])
            )

        # Pattern detection
        patterns = self.pattern_recognizer.detect_chart_patterns(df.tail(100))

        # Add pattern flags
        for pattern_name, pattern_data in patterns.items():
            if pattern_data.get('detected', False):
                df[f'pattern_{pattern_name}'] = 1
                df[f'pattern_{pattern_name}_confidence'] = pattern_data.get('confidence', 0)
            else:
                df[f'pattern_{pattern_name}'] = 0
                df[f'pattern_{pattern_name}_confidence'] = 0

        return df

    def _add_market_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        try:
            # Load S&P 500 data
            sp500 = web.DataReader('^spx', 'stooq', self.config.start_date, self.config.end_date)
            sp500 = sp500.sort_index()
            sp500['SP500_Return'] = sp500['Close'].pct_change()

            # Merge with stock data
            df = df.merge(sp500[['Close', 'SP500_Return']],
                          left_index=True, right_index=True,
                          how='left', suffixes=('', '_SP500'))

            # Beta calculation
            df['Beta'] = df['Return'].rolling(60).cov(df['SP500_Return']) / df['SP500_Return'].rolling(60).var()

            # Relative strength
            df['RS'] = df['Close'] / df['Close_SP500']
            df['RS_MA'] = df['RS'].rolling(20).mean()

        except Exception as e:
            logger.warning(f"Could not load market data: {e}")

        return df

    def prepare_sequences(self, df: pd.DataFrame, time_step: int = 60,
                          future_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        # Select features
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        features = df[feature_cols].values

        # Scale features
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(features)

        # Create sequences
        X, y = [], []
        for i in range(time_step, len(scaled_features) - future_steps + 1):
            X.append(scaled_features[i - time_step:i])
            # Target is next 'future_steps' closing prices
            y.append(df['Close'].iloc[i:i + future_steps].values)

        return np.array(X), np.array(y), scaler
