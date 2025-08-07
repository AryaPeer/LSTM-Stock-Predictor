from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import logging
from typing import Dict, Any

# Import configuration
from core.config import config

# Import services
from services.prediction_service import PredictionService
from services.market_analysis_service import MarketAnalysisService
from services.backtesting_service import BacktestingService

# Import repositories
from repositories.market_data_repository import MarketDataRepository
from repositories.model_repository import ModelRepository

# Import utilities
from utils.decorators import timer, validate_ticker
from utils.exceptions import (
    InvalidTickerError,
    DataNotFoundError,
    BacktestError
)

# Import backend modules
from backend.models import ModelType
from backend.data_pipeline import DataPipeline
from core.risk_manager import RiskManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Configure CORS
CORS(app, origins=['http://localhost:*', 'https://*.github.io'])

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[f"{config.rate_limit_per_hour} per hour",
                    f"{config.rate_limit_per_minute} per minute"]
)

# Configure caching
cache = Cache(app, config={
    'CACHE_TYPE': config.cache_type,
    'CACHE_DEFAULT_TIMEOUT': config.cache_timeout,
    'CACHE_THRESHOLD': config.cache_threshold
})

# Dependency Injection Container


class ServiceContainer:
    def __init__(self):
        # Initialize repositories
        self.market_data_repo = MarketDataRepository(config.cache_dir)
        self.model_repo = ModelRepository(config.model_dir)

        # Initialize data pipeline
        self.data_pipeline = DataPipeline()

        # Initialize services
        self.prediction_service = PredictionService(self.data_pipeline)
        self.market_analysis_service = MarketAnalysisService(self.data_pipeline)
        self.backtesting_service = BacktestingService(
            self.data_pipeline,
            self.prediction_service
        )

        # Initialize risk manager
        self.risk_manager = RiskManager(
            max_portfolio_risk=config.max_portfolio_risk,
            max_position_risk=config.max_position_risk
        )

# Initialize service container


services = ServiceContainer()


# API Response Handler
class APIResponse:
    @staticmethod
    def success(data: Any, message: str = "Success", meta: Dict = None) -> Dict:
        response = {
            "status": "success",
            "message": message,
            "data": data
        }
        if meta:
            response["meta"] = meta
        return jsonify(response)

    @staticmethod
    def error(message: str, code: int = 400, details: Any = None) -> tuple:
        response = {
            "status": "error",
            "message": message
        }
        if details:
            response["details"] = details
        return jsonify(response), code


# Health Check
@app.route('/api/health', methods=['GET'])
def health_check():
    return APIResponse.success({
        'status': 'healthy',
        'environment': config.env.value,
        'version': '2.0.0'
    })


# Prediction Endpoints
@app.route('/api/predictions', methods=['POST'])
@limiter.limit("10 per minute")
@timer
def create_prediction():
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()

        if not ticker:
            return APIResponse.error("Ticker symbol required", 400)

        model_type = ModelType(data.get('model_type', config.default_model))
        use_ensemble = data.get('use_ensemble', False)
        horizon = data.get('horizon', 10)

        result = services.prediction_service.predict(
            ticker=ticker,
            model_type=model_type,
            use_ensemble=use_ensemble,
            horizon=horizon
        )

        return APIResponse.success(result, "Prediction generated successfully")

    except InvalidTickerError as e:
        return APIResponse.error(str(e), 400)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return APIResponse.error("Prediction failed", 500, str(e))


# Market Analysis Endpoints
@app.route('/api/analysis/<ticker>', methods=['GET'])
@cache.cached(timeout=1800)
@validate_ticker
@timer
def get_market_analysis(ticker):
    try:
        result = services.market_analysis_service.analyze(ticker)
        return APIResponse.success(result, "Analysis completed")

    except DataNotFoundError as e:
        return APIResponse.error(str(e), 404)
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return APIResponse.error("Analysis failed", 500, str(e))


# Backtesting Endpoints
@app.route('/api/backtests', methods=['POST'])
@limiter.limit("5 per minute")
@timer
def run_backtest():
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()

        if not ticker:
            return APIResponse.error("Ticker symbol required", 400)

        strategy = data.get('strategy', 'technical')
        initial_capital = data.get('initial_capital', 100000)
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        result = services.backtesting_service.run_backtest(
            ticker=ticker,
            strategy=strategy,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date
        )

        return APIResponse.success(result, "Backtest completed")

    except BacktestError as e:
        return APIResponse.error(str(e), 400)
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        return APIResponse.error("Backtest failed", 500, str(e))


# Risk Management Endpoints
@app.route('/api/risk/position-size', methods=['POST'])
def calculate_position_size():
    try:
        data = request.json
        capital = data.get('capital', 100000)
        entry_price = data.get('entry_price')
        stop_loss_price = data.get('stop_loss_price')

        if not entry_price or not stop_loss_price:
            return APIResponse.error("Entry price and stop loss required", 400)

        position_size = services.risk_manager.calculate_position_size(
            capital=capital,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )

        risk_metrics = {
            'position_size': position_size,
            'position_value': position_size * entry_price,
            'max_risk': abs(entry_price - stop_loss_price) * position_size,
            'risk_percentage': abs(entry_price - stop_loss_price) / entry_price
        }

        return APIResponse.success(risk_metrics, "Position size calculated")

    except Exception as e:
        logger.error(f"Risk calculation error: {str(e)}")
        return APIResponse.error("Risk calculation failed", 500, str(e))


@app.route('/api/risk/portfolio', methods=['POST'])
def assess_portfolio_risk():
    try:
        data = request.json
        positions = data.get('positions', [])
        current_prices = data.get('current_prices', {})

        risk_assessment = services.risk_manager.assess_portfolio_risk(
            positions=positions,
            current_prices=current_prices
        )

        return APIResponse.success(risk_assessment, "Portfolio risk assessed")

    except Exception as e:
        logger.error(f"Portfolio risk error: {str(e)}")
        return APIResponse.error("Risk assessment failed", 500, str(e))


# Model Management Endpoints
@app.route('/api/models', methods=['GET'])
def list_models():
    try:
        ticker = request.args.get('ticker')
        models = services.model_repo.list_models(ticker)
        return APIResponse.success(models, f"Found {len(models)} models")

    except Exception as e:
        logger.error(f"Model listing error: {str(e)}")
        return APIResponse.error("Failed to list models", 500, str(e))


@app.route('/api/models/<model_id>', methods=['GET'])
def get_model(model_id):
    try:
        model_info = services.model_repo.get_model_info(model_id)
        if not model_info:
            return APIResponse.error("Model not found", 404)

        return APIResponse.success(model_info, "Model retrieved")

    except Exception as e:
        logger.error(f"Model retrieval error: {str(e)}")
        return APIResponse.error("Failed to retrieve model", 500, str(e))


@app.route('/api/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    try:
        success = services.model_repo.delete_model(model_id)
        if not success:
            return APIResponse.error("Model not found", 404)

        return APIResponse.success(None, "Model deleted")

    except Exception as e:
        logger.error(f"Model deletion error: {str(e)}")
        return APIResponse.error("Failed to delete model", 500, str(e))


# Static Files
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# Error Handlers
@app.errorhandler(404)
def not_found(e):
    return APIResponse.error("Endpoint not found", 404)


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return APIResponse.error("Internal server error", 500)


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return APIResponse.error("Rate limit exceeded", 429)


if __name__ == '__main__':
    logger.info(f"Starting application in {config.env.value} mode")
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug
    )
