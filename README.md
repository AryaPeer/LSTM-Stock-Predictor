# LSTM Stock Predictor

AI-powered stock prediction using CNN-BiLSTM with Attention mechanism.

## Features

- **Advanced AI Models**: CNN-BiLSTM-Attention, Transformer, WaveNet architectures
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages
- **Pattern Recognition**: Automatic chart pattern detection
- **Backtesting Engine**: Test strategies with historical data
- **Risk Management**: Position sizing and portfolio risk assessment
- **Web Interface**: Clean, responsive UI for stock analysis

## Installation

### Requirements
- Python 3.8 or higher
- TensorFlow 2.x
- Flask

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd LSTM-Stock-Predictor-main
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create necessary directories**
```bash
mkdir -p cache models
```

## Quick Start

### Run the application
```bash
python main.py
```

Or:
```bash
python app.py
```
## Usage

1. Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
2. Click "Analyze" to get predictions and technical analysis
3. View the 10-day price forecast and technical indicators

## API Endpoints

### Prediction
```http
POST /api/predictions
Content-Type: application/json

{
  "ticker": "AAPL",
  "horizon": 10
}
```

### Technical Analysis
```http
GET /api/analysis/{ticker}
```

### Backtesting
```http
POST /api/backtests
Content-Type: application/json

{
  "ticker": "AAPL",
  "strategy": "technical",
  "initial_capital": 100000
}
```

### Risk Management
```http
POST /api/risk/position-size
Content-Type: application/json

{
  "capital": 100000,
  "entry_price": 150,
  "stop_loss_price": 145
}
```

## Project Structure

```
LSTM-Stock-Predictor/
├── app.py                  # Flask API server
├── main.py                 # Application entry point
├── backend/                # ML models & data processing
│   ├── models.py           # CNN-BiLSTM-Attention, Transformer, WaveNet
│   ├── data_pipeline.py    # Feature engineering & data processing
│   └── backtesting.py      # Backtesting engine
├── services/               # Business logic layer
│   ├── prediction_service.py
│   ├── market_analysis_service.py
│   └── backtesting_service.py
├── repositories/           # Data access layer
│   ├── market_data_repository.py
│   └── model_repository.py
├── core/                   # Core utilities
│   ├── config.py           # Configuration management
│   └── risk_manager.py     # Risk management
├── static/                 # Web UI
│   ├── index.html
│   ├── script.js
│   └── styles.css
└── utils/                  # Helper utilities
    ├── decorators.py
    └── exceptions.py
```

## Technology Stack

- **Backend**: Flask, TensorFlow, Scikit-learn
- **Data Source**: Yahoo Finance (via yfinance)
- **Frontend**: HTML, JavaScript, Chart.js
- **ML Models**: CNN-BiLSTM with Attention, Transformer, WaveNet

## Model Performance

The CNN-BiLSTM-Attention model provides:
- Directional accuracy tracking
- RMSE, MAE, MAPE metrics
- Confidence intervals for predictions
- Sharpe ratio calculation

## Configuration

Environment variables:
- `PORT`: Server port (default: 5000)
- `HOST`: Server host (default: 0.0.0.0)
- `DEBUG`: Debug mode (default: false)
- `ENV`: Environment (development/staging/production)

## Features

### Technical Indicators
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- ATR (Average True Range)

### Pattern Recognition
- Support and Resistance levels
- Chart patterns detection
- Market regime identification
- Signal generation (buy/sell)

### Risk Metrics
- Position sizing calculation
- Portfolio risk assessment
- Stop-loss and take-profit levels
- Maximum drawdown analysis