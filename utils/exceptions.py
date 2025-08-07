class StockPredictorException(Exception):
    """Base exception for stock predictor application"""
    pass


class DataNotFoundError(StockPredictorException):
    """Raised when requested data is not available"""
    pass


class ModelNotFoundError(StockPredictorException):
    """Raised when requested model is not found"""
    pass


class InvalidTickerError(StockPredictorException):
    """Raised when ticker symbol is invalid"""
    pass


class InsufficientDataError(StockPredictorException):
    """Raised when there's not enough data for analysis"""
    pass


class BacktestError(StockPredictorException):
    """Raised when backtesting fails"""
    pass


class PredictionError(StockPredictorException):
    """Raised when prediction fails"""
    pass


class ConfigurationError(StockPredictorException):
    """Raised when configuration is invalid"""
    pass


class RiskLimitExceededError(StockPredictorException):
    """Raised when risk limits are exceeded"""
    pass
