# Backtesting Engine

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Position:
    symbol: str
    side: PositionSide
    entry_price: float
    entry_time: datetime
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0

    def close(self, exit_price: float, exit_time: datetime, fees: float = 0):
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.fees += fees

        if self.side == PositionSide.LONG:
            self.pnl = (exit_price - self.entry_price) * self.quantity - self.fees
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # SHORT
            self.pnl = (self.entry_price - exit_price) * self.quantity - self.fees
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def holding_period(self) -> Optional[timedelta]:
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None


@dataclass
class BacktestConfig:
    initial_capital: float = 100000
    position_size: float = 0.1  # Fraction of capital per trade
    max_positions: int = 5
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%

    # Risk management
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.05  # 5%
    max_drawdown_pct: float = 0.15  # 15%

    # Position sizing methods
    position_sizing: str = "fixed"  # fixed, kelly, risk_parity, vol_target
    kelly_fraction: float = 0.25  # Fractional Kelly
    target_volatility: float = 0.15  # 15% annual volatility

    # Rebalancing
    rebalance_frequency: str = "daily"  # daily, weekly, monthly

    # Performance calculation
    risk_free_rate: float = 0.02  # 2% annual
    benchmark: Optional[str] = "SPY"


class PerformanceMetrics:

    @staticmethod
    def calculate_metrics(returns: pd.Series, config: BacktestConfig) -> Dict[str, float]:
        metrics = {}

        # Basic metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)

        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = PerformanceMetrics.sharpe_ratio(returns, config.risk_free_rate)
        metrics['sortino_ratio'] = PerformanceMetrics.sortino_ratio(returns, config.risk_free_rate)
        metrics['calmar_ratio'] = PerformanceMetrics.calmar_ratio(returns)

        # Drawdown metrics
        drawdown_data = PerformanceMetrics.calculate_drawdowns(returns)
        metrics['max_drawdown'] = drawdown_data['max_drawdown']
        metrics['max_drawdown_duration'] = drawdown_data['max_duration']
        metrics['recovery_time'] = drawdown_data['recovery_time']

        # Risk metrics
        metrics['var_95'] = PerformanceMetrics.value_at_risk(returns, 0.95)
        metrics['cvar_95'] = PerformanceMetrics.conditional_value_at_risk(returns, 0.95)
        metrics['skewness'] = skew(returns)
        metrics['kurtosis'] = kurtosis(returns)

        # Win/loss metrics
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]

        metrics['win_rate'] = len(winning_returns) / len(returns) if len(returns) > 0 else 0
        metrics['avg_win'] = winning_returns.mean() if len(winning_returns) > 0 else 0
        metrics['avg_loss'] = losing_returns.mean() if len(losing_returns) > 0 else 0
        metrics['profit_factor'] = abs(winning_returns.sum() / losing_returns.sum()) if losing_returns.sum() != 0 else 0

        # Kelly criterion
        if metrics['win_rate'] > 0 and metrics['avg_loss'] != 0:
            metrics['kelly_fraction'] = (metrics['win_rate'] * metrics['avg_win'] +
                                         (1 - metrics['win_rate']) * metrics['avg_loss']) / metrics['avg_win']
        else:
            metrics['kelly_fraction'] = 0

        return metrics

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
        excess_returns = returns - risk_free_rate / 252
        if returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return 0

        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0

        return np.sqrt(252) * excess_returns.mean() / downside_std

    @staticmethod
    def calmar_ratio(returns: pd.Series) -> float:
        drawdown_data = PerformanceMetrics.calculate_drawdowns(returns)
        max_dd = drawdown_data['max_drawdown']

        if max_dd == 0:
            return 0

        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        return annual_return / abs(max_dd)

    @staticmethod
    def calculate_drawdowns(returns: pd.Series) -> Dict[str, Any]:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_drawdown = drawdown.min()

        # Find drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0

        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
                max_duration = max(max_duration, current_duration)
            else:
                drawdown_start = None
                current_duration = 0

        # Recovery time (time to recover from max drawdown)
        max_dd_idx = drawdown.idxmin()
        recovery_idx = cumulative[max_dd_idx:].ge(running_max[max_dd_idx]).idxmax() if max_dd_idx else None
        recovery_time = (recovery_idx - max_dd_idx).days if recovery_idx and max_dd_idx else None

        return {
            'max_drawdown': max_drawdown,
            'max_duration': max_duration,
            'recovery_time': recovery_time,
            'drawdown_series': drawdown
        }

    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float) -> float:
        return np.percentile(returns, (1 - confidence_level) * 100)

    @staticmethod
    def conditional_value_at_risk(returns: pd.Series, confidence_level: float) -> float:
        var = PerformanceMetrics.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()


class PositionSizer:

    @staticmethod
    def fixed_size(capital: float, config: BacktestConfig) -> float:
        return capital * config.position_size

    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float,
                        capital: float, fraction: float = 0.25) -> float:
        if avg_win == 0 or win_rate == 0:
            return 0

        kelly = (win_rate * avg_win + (1 - win_rate) * avg_loss) / avg_win
        kelly = max(0, min(kelly, 1))  # Bound between 0 and 1

        return capital * kelly * fraction

    @staticmethod
    def volatility_targeting(returns: pd.Series, target_vol: float,
                             capital: float, lookback: int = 60) -> float:
        if len(returns) < lookback:
            return capital * 0.1  # Default size

        recent_vol = returns.tail(lookback).std() * np.sqrt(252)
        if recent_vol == 0:
            return capital * 0.1

        position_size = (target_vol / recent_vol) * capital
        return min(position_size, capital)  # Don't use leverage

    @staticmethod
    def risk_parity(covariance_matrix: np.ndarray, capital: float) -> np.ndarray:
        n_assets = covariance_matrix.shape[0]

        def risk_contribution(weights, cov_matrix):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib

        def objective(weights, cov_matrix):
            contrib = risk_contribution(weights, cov_matrix)
            return np.sum((contrib - contrib.mean()) ** 2)

        initial_weights = np.ones(n_assets) / n_assets
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for _ in range(n_assets)]

        result = minimize(objective, initial_weights, args=(covariance_matrix,),
                          method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x * capital


class BacktestEngine:

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.positions: List[Position] = []
        self.open_positions: List[Position] = []
        self.capital = self.config.initial_capital
        self.equity_curve = []
        self.trades_log = []
        self.metrics = {}

    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting backtest...")

        # Initialize tracking
        portfolio_value = [self.capital]
        daily_returns = []

        for i in range(1, len(data)):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]

            # Check for signals
            if i < len(signals):
                signal = signals.iloc[i]

                # Entry signals
                if signal.get('buy', False) and len(self.open_positions) < self.config.max_positions:
                    self._enter_position(PositionSide.LONG, current_price, current_date)
                elif signal.get('sell', False) and len(self.open_positions) < self.config.max_positions:
                    self._enter_position(PositionSide.SHORT, current_price, current_date)

            # Check exit conditions
            self._check_exits(current_price, current_date)

            # Calculate portfolio value
            current_value = self._calculate_portfolio_value(current_price)
            portfolio_value.append(current_value)

            # Calculate daily return
            daily_return = (current_value - portfolio_value[-2]) / portfolio_value[-2]
            daily_returns.append(daily_return)

            # Risk management
            if self._check_risk_limits(current_value):
                self._close_all_positions(current_price, current_date)

        # Calculate final metrics
        self.equity_curve = pd.Series(portfolio_value, index=data.index)
        returns_series = pd.Series(daily_returns, index=data.index[1:])

        self.metrics = PerformanceMetrics.calculate_metrics(returns_series, self.config)
        self.metrics['total_trades'] = len(self.positions)
        self.metrics['open_positions'] = len(self.open_positions)
        self.metrics['final_capital'] = portfolio_value[-1]

        logger.info(f"Backtest complete. Final capital: ${portfolio_value[-1]:,.2f}")

        return {
            'metrics': self.metrics,
            'equity_curve': self.equity_curve,
            'trades': self._get_trades_summary(),
            'positions': self.positions
        }

    def _enter_position(self, side: PositionSide, price: float, time: datetime):
        # Calculate position size
        position_size = self._calculate_position_size()

        if position_size <= 0:
            return

        # Apply slippage
        entry_price = price * (1 + self.config.slippage if side == PositionSide.LONG else 1 - self.config.slippage)

        # Calculate quantity
        quantity = position_size / entry_price

        # Commission
        fees = position_size * self.config.commission

        # Create position
        position = Position(
            symbol="STOCK",
            side=side,
            entry_price=entry_price,
            entry_time=time,
            quantity=quantity,
            stop_loss=entry_price * (1 - self.config.stop_loss_pct) if side == PositionSide.LONG
            else entry_price * (1 + self.config.stop_loss_pct),
            take_profit=entry_price * (1 + self.config.take_profit_pct) if side == PositionSide.LONG
            else entry_price * (1 - self.config.take_profit_pct),
            fees=fees
        )

        self.positions.append(position)
        self.open_positions.append(position)
        self.capital -= (position_size + fees)

        logger.debug(f"Entered {side.value} position at {entry_price:.2f}")

    def _check_exits(self, price: float, time: datetime):
        positions_to_close = []

        for position in self.open_positions:
            should_exit = False

            # Check stop loss
            if position.side == PositionSide.LONG:
                if price <= position.stop_loss:
                    should_exit = True
                elif price >= position.take_profit:
                    should_exit = True
            else:  # SHORT
                if price >= position.stop_loss:
                    should_exit = True
                elif price <= position.take_profit:
                    should_exit = True

            if should_exit:
                positions_to_close.append(position)

        # Close positions
        for position in positions_to_close:
            self._close_position(position, price, time)

    def _close_position(self, position: Position, price: float, time: datetime):
        # Apply slippage
        exit_price = price * (1 - self.config.slippage if position.side == PositionSide.LONG
                              else 1 + self.config.slippage)

        # Commission
        fees = position.quantity * exit_price * self.config.commission

        # Close position
        position.close(exit_price, time, fees)

        # Update capital
        if position.side == PositionSide.LONG:
            self.capital += position.quantity * exit_price - fees
        else:  # SHORT
            self.capital += position.quantity * (2 * position.entry_price - exit_price) - fees

        # Remove from open positions
        self.open_positions.remove(position)

        logger.debug(f"Closed {position.side.value} position at {exit_price:.2f}, PnL: {position.pnl:.2f}")

    def _close_all_positions(self, price: float, time: datetime):
        for position in list(self.open_positions):
            self._close_position(position, price, time)

    def _calculate_portfolio_value(self, current_price: float) -> float:
        value = self.capital

        for position in self.open_positions:
            if position.side == PositionSide.LONG:
                value += position.quantity * current_price
            else:  # SHORT
                value += position.quantity * (2 * position.entry_price - current_price)

        return value

    def _calculate_position_size(self) -> float:
        if self.config.position_sizing == "fixed":
            return PositionSizer.fixed_size(self.capital, self.config)
        elif self.config.position_sizing == "kelly":
            # Use historical metrics for Kelly sizing
            if len(self.positions) > 10:
                wins = [p for p in self.positions if p.pnl > 0]
                losses = [p for p in self.positions if p.pnl < 0]

                if wins and losses:
                    win_rate = len(wins) / len(self.positions)
                    avg_win = np.mean([p.pnl_pct for p in wins])
                    avg_loss = np.mean([p.pnl_pct for p in losses])

                    return PositionSizer.kelly_criterion(win_rate, avg_win, abs(avg_loss),
                                                         self.capital, self.config.kelly_fraction)

            return PositionSizer.fixed_size(self.capital, self.config)
        elif self.config.position_sizing == "vol_target":
            if self.equity_curve:
                returns = self.equity_curve.pct_change().dropna()
                return PositionSizer.volatility_targeting(returns, self.config.target_volatility, self.capital)

            return PositionSizer.fixed_size(self.capital, self.config)
        else:
            return PositionSizer.fixed_size(self.capital, self.config)

    def _check_risk_limits(self, current_value: float) -> bool:
        drawdown = (current_value - self.config.initial_capital) / self.config.initial_capital

        if drawdown < -self.config.max_drawdown_pct:
            logger.warning(f"Max drawdown reached: {drawdown:.2%}")
            return True

        return False

    def _get_trades_summary(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame()

        trades_data = []
        for position in self.positions:
            trades_data.append({
                'entry_time': position.entry_time,
                'exit_time': position.exit_time,
                'side': position.side.value,
                'entry_price': position.entry_price,
                'exit_price': position.exit_price,
                'quantity': position.quantity,
                'pnl': position.pnl,
                'pnl_pct': position.pnl_pct,
                'fees': position.fees,
                'holding_period': position.holding_period.days if position.holding_period else None
            })

        return pd.DataFrame(trades_data)
