#!/usr/bin/env python3
"""
Baseline Models for Electricity Demand Forecasting
- Naive Forecast
- Seasonal Naive Forecast
- Moving Average
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class NaiveModel:
    """
    Naive forecast: Use the last observed value as the forecast
    """
    
    def __init__(self):
        self.name = "Naive"
        self.last_value = None
    
    def fit(self, train_data: pd.DataFrame):
        """Fit the naive model (just store last value)"""
        self.last_value = train_data['demand'].iloc[-1]
    
    def predict(self, horizon: int = 192) -> np.ndarray:
        """Generate naive forecast"""
        return np.full(horizon, self.last_value)


class SeasonalNaiveModel:
    """
    Seasonal Naive forecast: Use the value from the same time in the previous season
    For 15-minute data with daily seasonality: season = 96 (24 hours * 4)
    """
    
    def __init__(self, season_length: int = 96):
        self.name = "SeasonalNaive"
        self.season_length = season_length
        self.historical_data = None
    
    def fit(self, train_data: pd.DataFrame):
        """Fit the seasonal naive model"""
        self.historical_data = train_data['demand'].values
    
    def predict(self, horizon: int = 192) -> np.ndarray:
        """Generate seasonal naive forecast"""
        forecast = np.zeros(horizon)
        
        for i in range(horizon):
            # Use value from season_length steps ago
            lookback_idx = len(self.historical_data) - self.season_length + (i % self.season_length)
            if lookback_idx >= 0 and lookback_idx < len(self.historical_data):
                forecast[i] = self.historical_data[lookback_idx]
            else:
                # Fallback to last value if not enough history
                forecast[i] = self.historical_data[-1]
        
        return forecast


class MovingAverageModel:
    """
    Moving Average forecast: Use the average of the last N values
    """
    
    def __init__(self, window: int = 96):
        self.name = "MovingAverage"
        self.window = window
        self.historical_data = None
    
    def fit(self, train_data: pd.DataFrame):
        """Fit the moving average model"""
        self.historical_data = train_data['demand'].values
    
    def predict(self, horizon: int = 192) -> np.ndarray:
        """Generate moving average forecast"""
        # Use the average of the last window values
        last_values = self.historical_data[-self.window:]
        forecast_value = np.mean(last_values)
        return np.full(horizon, forecast_value)


def evaluate_baseline_model(model, train_data: pd.DataFrame, test_data: pd.DataFrame,
                            horizon: int = 192, stride: int = 96) -> Dict[str, float]:
    """
    Evaluate a baseline model with rolling forecasts
    
    Args:
        model: Baseline model instance
        train_data: Training data
        test_data: Test data
        horizon: Forecast horizon (192 = 48 hours)
        stride: Stride between forecasts (96 = 24 hours)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating {model.name} model...")
    
    # Fit model on training data
    model.fit(train_data)
    
    # Generate rolling forecasts on test data
    test_values = test_data['demand'].values
    all_predictions = []
    all_actuals = []
    
    # Start from the beginning of test data
    for start_idx in range(0, len(test_values) - horizon, stride):
        # Generate forecast
        forecast = model.predict(horizon)
        
        # Get actual values
        actual = test_values[start_idx:start_idx + horizon]
        
        # Store for evaluation (only last 96 steps = 24 hours)
        all_predictions.extend(forecast[-96:])
        all_actuals.extend(actual[-96:])
        
        # Update model with new data (for adaptive forecasting)
        # For simplicity, we'll keep the model fixed
    
    # Convert to arrays
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)
    
    # Ensure same length
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]
    
    # Calculate metrics
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    
    # Calculate MASE
    train_values = train_data['demand'].values
    naive_forecast = np.roll(train_values, -96)[:-96]
    mae_naive = np.mean(np.abs(train_values[96:] - naive_forecast))
    mase = mae / mae_naive if mae_naive > 0 else float('inf')
    
    results = {
        'mape': mape,
        'mae': mae,
        'rmse': rmse,
        'mase': mase,
        'crps': mae  # Simplified for deterministic forecasts
    }
    
    print(f"{model.name} Results:")
    print(f"  MAPE: {mape:.4f}%")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MASE: {mase:.4f}")
    
    return results


def create_baseline_models():
    """
    Factory function to create all baseline models
    """
    return {
        'Naive': NaiveModel(),
        'SeasonalNaive': SeasonalNaiveModel(season_length=96),  # Daily seasonality
        'MovingAverage': MovingAverageModel(window=96)  # 24-hour window
    }


if __name__ == "__main__":
    # Example usage
    models = create_baseline_models()
    print("Baseline models created:")
    for name, model in models.items():
        print(f"  - {name}")
