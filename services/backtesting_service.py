import pandas as pd
from typing import Dict, Any, Optional
import logging

from backend.backtesting import BacktestEngine, BacktestConfig
from backend.data_pipeline import DataPipeline
from services.prediction_service import PredictionService

logger = logging.getLogger(__name__)


class BacktestingService:
    def __init__(self, data_pipeline: DataPipeline = None,
                 prediction_service: PredictionService = None):
        self.data_pipeline = data_pipeline or DataPipeline()
        self.prediction_service = prediction_service or PredictionService(data_pipeline)

    def run_backtest(self,
                     ticker: str,
                     strategy: str = "ml_signals",
                     initial_capital: float = 100000,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Dict[str, Any]:

        logger.info(f"Running backtest for {ticker} with {strategy} strategy")

        # Load data
        df = self.data_pipeline.load_data(ticker)

        # Filter date range if specified
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        # Generate signals based on strategy
        signals = self._generate_signals(ticker, df, strategy)

        # Configure backtest
        config = BacktestConfig(
            initial_capital=initial_capital,
            position_size=0.1,
            max_positions=5,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            commission=0.001,
            slippage=0.0005
        )

        # Run backtest
        engine = BacktestEngine(config)
        results = engine.run_backtest(df, signals)

        # Format results
        return self._format_backtest_results(results, ticker, strategy, df)

    def _generate_signals(self, ticker: str, df: pd.DataFrame,
                          strategy: str) -> pd.DataFrame:
        if strategy == "ml_signals":
            return self._generate_ml_signals(ticker, df)
        elif strategy == "technical":
            return self._generate_technical_signals(df)
        elif strategy == "momentum":
            return self._generate_momentum_signals(df)
        elif strategy == "mean_reversion":
            return self._generate_mean_reversion_signals(df)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _generate_ml_signals(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        # Get ML predictions
        predictions = self.prediction_service.predict(ticker, horizon=5)

        signals = pd.DataFrame(index=df.index)
        signals['buy'] = False
        signals['sell'] = False

        # Generate signals based on predicted returns
        for i in range(len(df) - 5):
            if i < len(predictions['predictions']):
                pred_return = predictions['predictions'][0]['expected_return']

                if pred_return > 0.02:  # 2% expected gain
                    signals.iloc[i]['buy'] = True
                elif pred_return < -0.02:  # 2% expected loss
                    signals.iloc[i]['sell'] = True

        return signals

    def _generate_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index)
        signals['buy'] = False
        signals['sell'] = False

        # RSI signals
        signals.loc[df['RSI'] < 30, 'buy'] = True
        signals.loc[df['RSI'] > 70, 'sell'] = True

        # MACD crossover
        macd_cross_up = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
        macd_cross_down = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))

        signals.loc[macd_cross_up, 'buy'] = True
        signals.loc[macd_cross_down, 'sell'] = True

        return signals

    def _generate_momentum_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index)
        signals['buy'] = False
        signals['sell'] = False

        # Momentum based on rate of change
        roc = df['Close'].pct_change(periods=10)

        # Buy when momentum is strong positive
        signals.loc[roc > roc.quantile(0.8), 'buy'] = True

        # Sell when momentum turns negative
        signals.loc[roc < roc.quantile(0.2), 'sell'] = True

        return signals

    def _generate_mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index)
        signals['buy'] = False
        signals['sell'] = False

        # Z-score for mean reversion
        ma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        z_score = (df['Close'] - ma) / std

        # Buy when price is significantly below mean
        signals.loc[z_score < -2, 'buy'] = True

        # Sell when price is significantly above mean
        signals.loc[z_score > 2, 'sell'] = True

        return signals

    def _format_backtest_results(self, results: Dict, ticker: str,
                                 strategy: str, df: pd.DataFrame) -> Dict[str, Any]:
        metrics = results['metrics']

        # Calculate additional statistics
        winning_trades = len([t for t in results['positions'] if t.pnl > 0])
        losing_trades = len([t for t in results['positions'] if t.pnl < 0])
        total_trades = len(results['positions'])

        # Format response
        formatted = {
            'ticker': ticker,
            'strategy': strategy,
            'performance': {
                'total_return': metrics.get('total_return', 0),
                'annual_return': metrics.get('annual_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0)
            },
            'risk_metrics': {
                'volatility': metrics.get('volatility', 0),
                'var_95': metrics.get('var_95', 0),
                'cvar_95': metrics.get('cvar_95', 0),
                'skewness': metrics.get('skewness', 0),
                'kurtosis': metrics.get('kurtosis', 0)
            },
            'trade_statistics': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'profit_factor': metrics.get('profit_factor', 0),
                'avg_win': metrics.get('avg_win', 0),
                'avg_loss': metrics.get('avg_loss', 0)
            },
            'equity_curve': results['equity_curve'].tolist() if 'equity_curve' in results else [],
            'benchmark_comparison': {
                'strategy_return': metrics.get('total_return', 0),
                'buy_hold_return': float((df['Close'].iloc[-1] - df['Close'].iloc[0]) /
                                         df['Close'].iloc[0]),
                'excess_return': (metrics.get('total_return', 0) -
                                  float((df['Close'].iloc[-1] - df['Close'].iloc[0]) /
                                        df['Close'].iloc[0]))
            }
        }

        return formatted
