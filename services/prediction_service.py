import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta
import logging

from backend.models import ModelFactory, ModelConfig, ModelType
from backend.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, data_pipeline: DataPipeline = None):
        self.data_pipeline = data_pipeline or DataPipeline()
        self._model_cache = {}

    def predict(self,
                ticker: str,
                model_type: ModelType = ModelType.CNN_BILSTM_ATTENTION,
                use_ensemble: bool = False,
                horizon: int = 10) -> Dict[str, Any]:

        logger.info(f"Generating {horizon}-day prediction for {ticker}")

        # Load and prepare data
        df = self.data_pipeline.load_data(ticker)
        X, y, scaler = self.data_pipeline.prepare_sequences(df, future_steps=horizon)

        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Get or create model
        model = self._get_or_create_model(model_type, use_ensemble, X_train.shape[1:], horizon)

        # Train model
        if use_ensemble:
            model.train_ensemble(X_train, y_train, X_test, y_test)
            predictions = model.predict_ensemble(X_test[-1:])
        else:
            model.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                batch_size=32,
                epochs=100,
                callbacks=model.get_callbacks(),
                verbose=0
            )
            predictions = model.model.predict(X_test[-1:], verbose=0)

        # Calculate metrics
        val_predictions = (model.model.predict(X_test, verbose=0) if not use_ensemble
                           else model.predict_ensemble(X_test))
        metrics = self._calculate_metrics(val_predictions, y_test, df)

        # Format response
        return self._format_prediction_response(
            ticker, df, predictions[0], metrics, model_type, horizon
        )

    def _get_or_create_model(self, model_type: ModelType, use_ensemble: bool,
                             input_shape: tuple, horizon: int):
        cache_key = f"{model_type.value}_{use_ensemble}_{input_shape}_{horizon}"

        if cache_key not in self._model_cache:
            config = ModelConfig(
                model_type=model_type,
                use_ensemble=use_ensemble,
                future_steps=horizon
            )
            model = ModelFactory.create_model(config)

            if use_ensemble:
                model.build_ensemble(input_shape)
            else:
                model.build_model(input_shape)
                model.compile_model()

            self._model_cache[cache_key] = model

        return self._model_cache[cache_key]

    def _calculate_metrics(self, predictions: np.ndarray, actuals: np.ndarray,
                           df: pd.DataFrame) -> Dict[str, float]:
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        # Directional accuracy
        pred_direction = np.diff(predictions.flatten()) > 0
        actual_direction = np.diff(actuals.flatten()) > 0
        directional_accuracy = np.mean(pred_direction == actual_direction)

        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'directional_accuracy': float(directional_accuracy),
            'sharpe_ratio': self._calculate_sharpe(predictions, actuals)
        }

    def _calculate_sharpe(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        # Calculate returns from predictions
        pred_returns = np.diff(predictions.flatten()) / predictions.flatten()[:-1]
        if len(pred_returns) > 0 and np.std(pred_returns) > 0:
            return np.sqrt(252) * np.mean(pred_returns) / np.std(pred_returns)
        return 0.0

    def _format_prediction_response(self, ticker: str, df: pd.DataFrame,
                                    predictions: np.ndarray, metrics: Dict,
                                    model_type: ModelType, horizon: int) -> Dict[str, Any]:
        last_date = df.index[-1]
        pred_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq='B')

        # Calculate confidence intervals
        prediction_std = np.std(predictions)

        return {
            'ticker': ticker,
            'model_type': model_type.value,
            'current_price': float(df['Close'].iloc[-1]),
            'predictions': [
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'price': float(pred),
                    'confidence_interval': {
                        'lower': float(pred - 1.96 * prediction_std),
                        'upper': float(pred + 1.96 * prediction_std)
                    },
                    'expected_return': float((pred - df['Close'].iloc[-1]) /
                                             df['Close'].iloc[-1])
                } for date, pred in zip(pred_dates, predictions)
            ],
            'metrics': metrics,
            'metadata': {
                'horizon_days': horizon,
                'training_samples': len(df),
                'last_update': datetime.now().isoformat()
            }
        }
