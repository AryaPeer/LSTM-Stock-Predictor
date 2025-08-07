# Project Structure Guide

## Overview
This is a production-ready stock prediction platform implementing CNN-BiLSTM with Attention architecture for time series forecasting.

## Directory Structure

### Root Level
- `app.py` - Flask application server with dependency injection
- `main.py` - Application entry point
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `.gitignore` - Git ignore configuration

### `/core` - Core Utilities
- `config.py` - Centralized configuration management
- `risk_manager.py` - Position sizing and portfolio risk calculations
- `__init__.py` - Module initialization

### `/backend` - ML Models & Data Processing
- `models.py` - CNN-BiLSTM-Attention, Transformer, WaveNet implementations
- `data_pipeline.py` - Feature engineering and data preprocessing
- `backtesting.py` - Backtesting engine with performance metrics
- `__init__.py` - Module initialization

### `/services` - Business Logic Layer
- `prediction_service.py` - ML prediction orchestration
- `market_analysis_service.py` - Technical analysis and pattern detection
- `backtesting_service.py` - Strategy backtesting service
- `__init__.py` - Module initialization

### `/repositories` - Data Access Layer
- `market_data_repository.py` - Yahoo Finance data fetching with caching
- `model_repository.py` - Model persistence and retrieval
- `__init__.py` - Module initialization

### `/utils` - Helper Utilities
- `exceptions.py` - Custom exception definitions
- `decorators.py` - Timer and validation decorators
- `__init__.py` - Module initialization

### `/static` - Web UI
- `index.html` - Frontend interface
- `script.js` - JavaScript for API interaction and charts
- `styles.css` - CSS styling

## Design Patterns Used

1. **Service Layer Pattern** - Clean separation of business logic
2. **Repository Pattern** - Abstract data access
3. **Factory Pattern** - Model creation (ModelFactory)
4. **Dependency Injection** - ServiceContainer in app.py
5. **Decorator Pattern** - Cross-cutting concerns (timing, validation)

## Key Features

### Machine Learning
- CNN-BiLSTM with Attention mechanism
- Transformer models for sequence modeling
- WaveNet for temporal patterns
- Ensemble learning capabilities
- Comprehensive feature engineering

### Technical Analysis
- RSI, MACD, Bollinger Bands
- Moving averages (SMA, EMA)
- Support/resistance detection
- Pattern recognition
- Market regime identification

### Risk Management
- Position sizing algorithms
- Portfolio risk assessment
- Stop-loss/take-profit calculations
- Maximum drawdown analysis

### Production Features
- API rate limiting
- Response caching
- Comprehensive error handling
- Logging system
- Configuration management

## API Endpoints

### Predictions
- `POST /api/predictions` - Generate stock predictions
  ```json
  {
    "ticker": "AAPL",
    "horizon": 10
  }
  ```

### Analysis
- `GET /api/analysis/<ticker>` - Get technical analysis

### Backtesting
- `POST /api/backtests` - Run strategy backtest
  ```json
  {
    "ticker": "AAPL",
    "strategy": "technical",
    "initial_capital": 100000
  }
  ```

### Risk Management
- `POST /api/risk/position-size` - Calculate position size
- `POST /api/risk/portfolio` - Assess portfolio risk

### Model Management
- `GET /api/models` - List saved models
- `GET /api/models/<model_id>` - Get model details
- `DELETE /api/models/<model_id>` - Delete model

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p cache models

# Run the application
python main.py
# OR
python app.py

# Access at http://localhost:5000
```

## Technology Stack

- **Backend**: Flask, TensorFlow, Scikit-learn, Pandas, NumPy
- **Data Source**: Yahoo Finance (yfinance)
- **Frontend**: HTML5, JavaScript (ES6), Chart.js
- **ML Models**: Deep learning with attention mechanisms
- **Caching**: In-memory caching with Flask-Caching

## Code Organization

The codebase follows clean architecture principles:
- Clear separation of concerns
- Dependency injection for testability
- Service layer for business logic
- Repository pattern for data access
- Proper error handling throughout
- Consistent naming conventions

## Performance Optimizations

- Caching of API responses
- Rate limiting for API protection
- Efficient data pipeline with vectorized operations
- Model caching to avoid retraining
- Optimized feature engineering

## File Count
- **19 Python files** - Backend implementation
- **3 Frontend files** - Web UI
- **3 Documentation files** - README, PROJECT_STRUCTURE, .gitignore
- **1 Requirements file** - Python dependencies
- **Total: 26 clean, essential files**

## Notes

- All code is production-ready and follows best practices
- No dead code or unused files
- Clean dependency structure
- Proper error handling and logging throughout
- Suitable for deployment to cloud platforms