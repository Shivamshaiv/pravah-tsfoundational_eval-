#!/usr/bin/env python3
"""
Foundation Models for Time Series Forecasting
- TimesFM (Google)
- Moirai (Salesforce)
- Chronos (Amazon)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Import GPU utilities
from models.gpu_utils import get_gpu_config, is_gpu_available, get_device


class FoundationModelWrapper:
    """
    Wrapper for foundation models (zero-shot forecasting)
    """
    
    def __init__(self, model_name: str, model_config: Optional[Dict] = None):
        self.model_name = model_name
        self.model_config = model_config or {}
        self.model = None
        self.is_loaded = False
        self.gpu_config = get_gpu_config()
        self.device = get_device()
        
    def load_model(self):
        """Load the foundation model"""
        
        print(f"Loading {self.model_name} model...")
        print(f"Using device: {self.device} ({self.gpu_config.device_name})")
        
        try:
            if self.model_name == "TimesFM":
                self._load_timesfm()
            elif self.model_name == "Moirai":
                self._load_moirai()
            elif self.model_name == "Chronos":
                self._load_chronos()
            else:
                raise ValueError(f"Unknown foundation model: {self.model_name}")
            
            self.is_loaded = True
            print(f"{self.model_name} loaded successfully!")
            
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            print(f"Skipping {self.model_name} - model not available")
            self.is_loaded = False
    
    def _load_timesfm(self):
        """Load TimesFM model"""
        try:
            import timesfm
            backend = 'gpu' if is_gpu_available() else 'cpu'
            print(f"TimesFM backend: {backend}")
            self.model = timesfm.TimesFm(
                context_len=512,
                horizon_len=192,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                backend=backend
            )
            self.model.load_from_checkpoint()
        except ImportError:
            print("TimesFM not installed. Install with: pip install timesfm")
            raise
    
    def _load_moirai(self):
        """Load Moirai model"""
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            import torch
            
            print(f"Loading Moirai to device: {self.device}")
            
            # Load Moirai-1.0-R-Small (can also use Medium or Large)
            self.model = MoiraiForecast.load_from_checkpoint(
                checkpoint_path="Salesforce/moirai-1.0-R-small",
                prediction_length=192,
                context_length=512,
                patch_size=32,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except ImportError:
            print("Moirai not installed. Install with: pip install uni2ts")
            raise
    
    def _load_chronos(self):
        """Load Chronos model"""
        try:
            from chronos import ChronosPipeline
            import torch
            
            device_map = 'cuda' if is_gpu_available() else 'cpu'
            print(f"Loading Chronos to device: {device_map}")
            
            # Load Chronos-T5-Small (can also use Base, Large)
            self.model = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )
            
        except ImportError:
            print("Chronos not installed. Install with: pip install chronos-forecasting")
            raise
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available (deprecated, use is_gpu_available)"""
        return is_gpu_available()
    
    def predict(self, history: np.ndarray, horizon: int = 192) -> np.ndarray:
        """Generate predictions using the foundation model"""
        
        if not self.is_loaded:
            raise ValueError(f"{self.model_name} model not loaded")
        
        if self.model_name == "TimesFM":
            return self._predict_timesfm(history, horizon)
        elif self.model_name == "Moirai":
            return self._predict_moirai(history, horizon)
        elif self.model_name == "Chronos":
            return self._predict_chronos(history, horizon)
    
    def _predict_timesfm(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """TimesFM prediction"""
        # Use last 512 points as context
        context = history[-512:] if len(history) > 512 else history
        
        forecast = self.model.forecast(
            inputs=[context],
            freq=[0],  # 15-minute frequency
        )
        
        return forecast[0, :horizon]
    
    def _predict_moirai(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Moirai prediction"""
        import torch
        
        # Use last 512 points as context
        context = history[-512:] if len(history) > 512 else history
        context_tensor = torch.tensor(context).unsqueeze(0).unsqueeze(-1)
        
        # Move to device
        context_tensor = context_tensor.to(self.device)
        
        with torch.no_grad():
            forecast = self.model.forward(
                past_target=context_tensor,
                past_observed_target=torch.ones_like(context_tensor),
            )
        
        # Get median forecast
        forecast_samples = forecast.samples.cpu().numpy()
        forecast_median = np.median(forecast_samples, axis=1).squeeze()
        
        return forecast_median[:horizon]
    
    def _predict_chronos(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Chronos prediction"""
        import torch
        
        # Use last 512 points as context
        context = history[-512:] if len(history) > 512 else history
        context_tensor = torch.tensor(context).unsqueeze(0)
        
        forecast = self.model.predict(
            context=context_tensor,
            prediction_length=horizon,
            num_samples=20,
        )
        
        # Get median forecast
        forecast_median = torch.median(forecast, dim=1).values.squeeze().cpu().numpy()
        
        return forecast_median
    
    def evaluate(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame,
                 train_features: Optional[pd.DataFrame] = None,
                 val_features: Optional[pd.DataFrame] = None,
                 test_features: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Complete evaluation pipeline for foundation models"""
        
        print(f"=" * 50)
        print(f"{self.model_name} MODEL EVALUATION")
        print("=" * 50)
        
        # Load model
        self.load_model()
        
        if not self.is_loaded:
            return {
                'mape': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'mase': float('inf'),
                'crps': float('inf')
            }
        
        # Combine train and val for context
        full_history = pd.concat([train_data, val_data])
        history_values = full_history['demand'].values
        
        # Generate predictions for test set
        test_values = test_data['demand'].values
        predictions = []
        
        # Rolling forecast with 24-hour stride
        stride = 96  # 24 hours in 15-min intervals
        horizon = 192  # 48 hours
        
        for i in range(0, len(test_values) - horizon, stride):
            # Get history up to this point
            context_end = len(history_values) + i
            context = np.concatenate([history_values, test_values[:i]]) if i > 0 else history_values
            
            # Predict next 48 hours
            forecast = self.predict(context, horizon=horizon)
            
            # Take last 24 hours (96 points)
            predictions.extend(forecast[-96:])
        
        predictions = np.array(predictions)
        actual_values = test_values[:len(predictions)]
        
        # Calculate metrics
        mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
        mae = np.mean(np.abs(actual_values - predictions))
        rmse = np.sqrt(np.mean((actual_values - predictions) ** 2))
        
        # MASE
        train_values = train_data['demand'].values
        naive_forecast = np.roll(train_values, -96)[:-96]
        mae_naive = np.mean(np.abs(train_values[96:] - naive_forecast))
        mase = mae / mae_naive if mae_naive > 0 else float('inf')
        
        results = {
            'mape': mape,
            'mae': mae,
            'rmse': rmse,
            'mase': mase,
            'crps': mae
        }
        
        print(f"{self.model_name} Results:")
        print(f"  MAPE: {mape:.4f}%")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MASE: {mase:.4f}")
        
        return results


def create_foundation_models():
    """Factory function to create all foundation models"""
    return {
        'TimesFM': FoundationModelWrapper('TimesFM'),
        'Moirai': FoundationModelWrapper('Moirai'),
        'Chronos': FoundationModelWrapper('Chronos')
    }


if __name__ == "__main__":
    models = create_foundation_models()
    print("Foundation models created:")
    for name in models.keys():
        print(f"  - {name}")
