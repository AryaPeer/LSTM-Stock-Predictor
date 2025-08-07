import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    position_size: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    max_risk_per_trade: float
    portfolio_heat: float


class RiskManager:
    def __init__(self,
                 max_portfolio_risk: float = 0.02,
                 max_position_risk: float = 0.01,
                 max_positions: int = 5):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_positions = max_positions

    def calculate_position_size(self,
                                capital: float,
                                entry_price: float,
                                stop_loss_price: float,
                                risk_per_trade: Optional[float] = None) -> float:

        if risk_per_trade is None:
            risk_per_trade = self.max_position_risk

        # Calculate risk amount
        risk_amount = capital * risk_per_trade

        # Calculate position size based on stop loss
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk == 0:
            return 0

        shares = risk_amount / price_risk
        position_value = shares * entry_price

        # Limit position size to max percentage of capital
        max_position_value = capital * 0.25  # Max 25% per position
        if position_value > max_position_value:
            position_value = max_position_value
            shares = position_value / entry_price

        return shares

    def calculate_kelly_criterion(self,
                                  win_rate: float,
                                  avg_win: float,
                                  avg_loss: float,
                                  kelly_fraction: float = 0.25) -> float:

        if avg_loss == 0 or win_rate == 0:
            return 0

        # Kelly formula: f = (p*b - q) / b
        # where p = win_rate, q = loss_rate, b = win/loss ratio
        b = avg_win / abs(avg_loss)
        q = 1 - win_rate

        kelly = (win_rate * b - q) / b

        # Apply Kelly fraction (never use full Kelly)
        kelly = kelly * kelly_fraction

        # Limit to reasonable bounds
        return max(0, min(kelly, 0.25))

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        # Value at Risk calculation
        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        # Conditional Value at Risk (Expected Shortfall)
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        excess_returns = returns - risk_free_rate / 252
        if returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0

        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

    def calculate_max_drawdown(self, prices: pd.Series) -> Tuple[float, int]:
        cummax = prices.expanding().max()
        drawdown = (prices - cummax) / cummax
        max_dd = drawdown.min()

        # Calculate drawdown duration
        dd_start = drawdown.idxmin()
        dd_end = prices[dd_start:].ge(cummax[dd_start]).idxmax() if dd_start else None
        duration = (dd_end - dd_start).days if dd_end and dd_start else 0

        return max_dd, duration

    def assess_portfolio_risk(self, positions: list, current_prices: Dict[str, float]) -> Dict[str, Any]:
        total_exposure = 0
        total_risk = 0

        for position in positions:
            ticker = position['ticker']
            shares = position['shares']
            entry_price = position['entry_price']
            stop_loss = position.get('stop_loss', entry_price * 0.98)

            current_price = current_prices.get(ticker, entry_price)
            position_value = shares * current_price
            position_risk = shares * (current_price - stop_loss)

            total_exposure += position_value
            total_risk += position_risk

        return {
            'total_exposure': total_exposure,
            'total_risk': total_risk,
            'risk_percentage': total_risk / total_exposure if total_exposure > 0 else 0,
            'position_count': len(positions),
            'concentration_risk': self._calculate_concentration_risk(positions, current_prices)
        }

    def _calculate_concentration_risk(self, positions: list, current_prices: Dict[str, float]) -> float:
        if not positions:
            return 0

        position_values = []
        for position in positions:
            ticker = position['ticker']
            shares = position['shares']
            current_price = current_prices.get(ticker, position['entry_price'])
            position_values.append(shares * current_price)

        total_value = sum(position_values)
        if total_value == 0:
            return 0

        # Calculate Herfindahl index for concentration
        concentrations = [(pv / total_value) ** 2 for pv in position_values]
        return sum(concentrations)

    def get_risk_limits(self, capital: float) -> Dict[str, float]:
        return {
            'max_position_size': capital * 0.25,
            'max_position_risk': capital * self.max_position_risk,
            'max_portfolio_risk': capital * self.max_portfolio_risk,
            'max_daily_loss': capital * 0.05,
            'max_positions': self.max_positions
        }
