import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

from backend.data_pipeline import DataPipeline, PatternRecognizer

logger = logging.getLogger(__name__)


class MarketAnalysisService:
    def __init__(self, data_pipeline: DataPipeline = None):
        self.data_pipeline = data_pipeline or DataPipeline()
        self.pattern_recognizer = PatternRecognizer()

    def analyze(self, ticker: str) -> Dict[str, Any]:
        logger.info(f"Performing comprehensive analysis for {ticker}")

        df = self.data_pipeline.load_data(ticker)

        analysis = {
            'ticker': ticker,
            'current_price': float(df['Close'].iloc[-1]),
            'technical_indicators': self._get_technical_indicators(df),
            'patterns': self._detect_patterns(df),
            'support_resistance': self._get_support_resistance(df),
            'market_regime': self._identify_market_regime(df),
            'signals': self._generate_signals(df),
            'risk_metrics': self._calculate_risk_metrics(df)
        }

        return analysis

    def _get_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        latest = df.iloc[-1]

        indicators = {
            'trend': {
                'sma_20': float(latest.get('MA_20', 0)),
                'sma_50': float(latest.get('MA_50', 0)),
                'sma_200': float(latest.get('MA_200', 0)),
                'ema_12': float(latest.get('EMA_12', 0)),
                'ema_26': float(latest.get('EMA_26', 0))
            },
            'momentum': {
                'rsi': float(latest.get('RSI', 50)),
                'macd': float(latest.get('MACD', 0)),
                'macd_signal': float(latest.get('MACD_signal', 0)),
                'macd_histogram': float(latest.get('MACD_diff', 0)),
                'stochastic_k': float(latest.get('Stoch_k', 0)),
                'stochastic_d': float(latest.get('Stoch_d', 0))
            },
            'volatility': {
                'atr': float(latest.get('ATR', 0)),
                'bollinger_upper': float(latest.get('BB_upper', 0)),
                'bollinger_middle': float(latest.get('BB_middle', 0)),
                'bollinger_lower': float(latest.get('BB_lower', 0)),
                'bollinger_width': float(latest.get('BB_width', 0))
            },
            'volume': {
                'volume': float(latest.get('Volume', 0)),
                'obv': float(latest.get('OBV', 0)),
                'mfi': float(latest.get('MFI', 0))
            }
        }

        return indicators

    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        patterns = self.pattern_recognizer.detect_chart_patterns(df)

        detected = []
        for pattern_name, pattern_data in patterns.items():
            if pattern_data.get('detected', False):
                detected.append({
                    'name': pattern_name.replace('_', ' ').title(),
                    'type': pattern_data.get('type', 'unknown'),
                    'confidence': pattern_data.get('confidence', 0),
                    'details': pattern_data
                })

        return {
            'detected_patterns': detected,
            'pattern_count': len(detected)
        }

    def _get_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        levels = self.pattern_recognizer.detect_support_resistance(df['Close'])
        current_price = df['Close'].iloc[-1]

        # Find nearest levels
        nearest_support = None
        nearest_resistance = None

        for support in levels['support']:
            if support < current_price:
                if nearest_support is None or support > nearest_support:
                    nearest_support = support

        for resistance in levels['resistance']:
            if resistance > current_price:
                if nearest_resistance is None or resistance < nearest_resistance:
                    nearest_resistance = resistance

        return {
            'support_levels': levels['support'],
            'resistance_levels': levels['resistance'],
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'distance_to_support': (float((current_price - nearest_support) / current_price * 100)
                                    if nearest_support else None),
            'distance_to_resistance': (float((nearest_resistance - current_price) / current_price * 100)
                                       if nearest_resistance else None)
        }

    def _identify_market_regime(self, df: pd.DataFrame) -> Dict[str, str]:
        latest = df.iloc[-1]

        # Volatility regime
        vol_20 = latest.get('Volatility_20', 0)
        vol_regime = 'high' if vol_20 > 0.03 else 'low' if vol_20 < 0.01 else 'normal'

        # Trend regime
        ma20 = latest.get('MA_20', 0)
        ma50 = latest.get('MA_50', 0)
        ma200 = latest.get('MA_200', 0)
        close = latest['Close']

        if close > ma20 > ma50 > ma200:
            trend_regime = 'strong_uptrend'
        elif close < ma20 < ma50 < ma200:
            trend_regime = 'strong_downtrend'
        elif close > ma50:
            trend_regime = 'uptrend'
        elif close < ma50:
            trend_regime = 'downtrend'
        else:
            trend_regime = 'sideways'

        # Volume regime
        volume = latest.get('Volume', 0)
        avg_volume = df['Volume'].tail(20).mean()
        volume_regime = 'high' if volume > avg_volume * 1.5 else 'low' if volume < avg_volume * 0.5 else 'normal'

        return {
            'volatility': vol_regime,
            'trend': trend_regime,
            'volume': volume_regime,
            'overall': self._determine_overall_regime(vol_regime, trend_regime, volume_regime)
        }

    def _determine_overall_regime(self, vol: str, trend: str, volume: str) -> str:
        if 'strong' in trend and vol == 'low':
            return 'trending'
        elif vol == 'high':
            return 'volatile'
        elif trend == 'sideways':
            return 'ranging'
        else:
            return 'normal'

    def _generate_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        signals = []
        latest = df.iloc[-1]

        # RSI signals
        rsi = latest.get('RSI', 50)
        if rsi < 30:
            signals.append({
                'type': 'BUY',
                'strength': 'STRONG',
                'indicator': 'RSI',
                'reason': f'Oversold (RSI={rsi:.1f})'
            })
        elif rsi > 70:
            signals.append({
                'type': 'SELL',
                'strength': 'STRONG',
                'indicator': 'RSI',
                'reason': f'Overbought (RSI={rsi:.1f})'
            })

        # MACD signals
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_signal', 0)
        prev_macd = df['MACD'].iloc[-2] if len(df) > 1 else 0
        prev_signal = df['MACD_signal'].iloc[-2] if len(df) > 1 else 0

        if macd > macd_signal and prev_macd <= prev_signal:
            signals.append({
                'type': 'BUY',
                'strength': 'MEDIUM',
                'indicator': 'MACD',
                'reason': 'Bullish crossover'
            })
        elif macd < macd_signal and prev_macd >= prev_signal:
            signals.append({
                'type': 'SELL',
                'strength': 'MEDIUM',
                'indicator': 'MACD',
                'reason': 'Bearish crossover'
            })

        # Moving average signals
        close = latest['Close']
        ma20 = latest.get('MA_20', close)

        if df['Close'].iloc[-2] <= df['MA_20'].iloc[-2] and close > ma20:
            signals.append({
                'type': 'BUY',
                'strength': 'MEDIUM',
                'indicator': 'MA',
                'reason': 'Price crossed above MA20'
            })

        return signals

    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        returns = df['Return'].dropna()

        if len(returns) < 20:
            return {}

        # Calculate various risk metrics
        metrics = {
            'volatility_daily': float(returns.std()),
            'volatility_annual': float(returns.std() * np.sqrt(252)),
            'var_95': float(np.percentile(returns, 5)),
            'cvar_95': float(returns[returns <= np.percentile(returns, 5)].mean()),
            'max_drawdown': float(self._calculate_max_drawdown(df['Close'])),
            'beta': float(df.get('Beta', [1.0]).iloc[-1]) if 'Beta' in df.columns else 1.0,
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis())
        }

        return metrics

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        cummax = prices.expanding().max()
        drawdown = (prices - cummax) / cummax
        return drawdown.min()
