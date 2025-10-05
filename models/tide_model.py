#!/usr/bin/env python3
"""
TiDE Model Implementation for Electricity Demand Forecasting

Based on the TiDE model from brpl_operational_model.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from darts import TimeSeries
from darts.models import TiDEModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae, rmse
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Import GPU utilities
from models.gpu_utils import get_gpu_config, is_gpu_available, get_trainer_kwargs, optimize_batch_size


class CustomLast24hMAELoss(nn.Module):
    """Custom loss function focusing on the last 24 hours of 48-hour forecast"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        # For 15-minute data, last 24h = 96 time steps (24 * 4)
        return torch.mean(torch.abs(y_true[:, -96:, :] - y_pred[:, -96:, :]))


class TiDEModelWrapper:
    """
    Wrapper for TiDE model with electricity demand forecasting specific configuration
    """
    
    def __init__(self, model_config: Optional[Dict] = None):
        """
        Initialize TiDE model with default or custom configuration
        
        Args:
            model_config: Dictionary with model hyperparameters
        """
        self.model_config = model_config or self._get_default_config()
        self.model = None
        self.scaler_y = None
        self.scaler_features = None
        self.is_trained = False
        self.gpu_config = get_gpu_config()
        
    def _get_default_config(self) -> Dict:
        """Get default model configuration"""
        return {
            'input_chunk_length': 192,      # 48 hours of input (192 * 15min)
            'output_chunk_length': 192,     # 48 hours of output (192 * 15min)
            'hidden_size': 1024,
            'temporal_decoder_hidden': 32,
            'temporal_width_past': 4,
            'temporal_width_future': 4,
            'num_encoder_layers': 5,
            'num_decoder_layers': 5,
            'decoder_output_dim': 32,
            'use_layer_norm': False,
            'dropout': 0.2,
            'n_epochs': 50,
            'batch_size': 64,
            'learning_rate': 5e-4,
            'patience': 10,
            'use_custom_loss': True,
            'random_state': 42
        }
    
    def create_model(self, target_dim: int = 1, feature_dim: int = 0) -> TiDEModel:
        """
        Create TiDE model instance with configured parameters
        """
        config = self.model_config
        
        # Optimize batch size for device
        batch_size = optimize_batch_size(config['batch_size'])
        config['batch_size'] = batch_size
        
        print(f"Creating TiDE model")
        print(f"Device: {self.gpu_config.device} ({self.gpu_config.device_name})")
        print(f"Batch size: {batch_size}")
        
        # Create loss function
        loss_fn = CustomLast24hMAELoss() if config['use_custom_loss'] else None
        
        # Get trainer kwargs optimized for device
        trainer_kwargs = get_trainer_kwargs(
            gradient_clip_val=0.5,
            precision="32-true",
            callbacks=[EarlyStopping(monitor="val_loss", patience=config['patience'])]
        )
        
        # Create model
        model = TiDEModel(
            input_chunk_length=config['input_chunk_length'],
            output_chunk_length=config['output_chunk_length'],
            hidden_size=config['hidden_size'],
            temporal_decoder_hidden=config['temporal_decoder_hidden'],
            temporal_width_past=config['temporal_width_past'],
            temporal_width_future=config['temporal_width_future'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            decoder_output_dim=config['decoder_output_dim'],
            use_layer_norm=config['use_layer_norm'],
            dropout=config['dropout'],
            n_epochs=config['n_epochs'],
            batch_size=batch_size,
            optimizer_kwargs={"lr": config['learning_rate']},
            loss_fn=loss_fn,
            pl_trainer_kwargs=trainer_kwargs,
            save_checkpoints=True,
            force_reset=True,
            random_state=config['random_state']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                    train_features: pd.DataFrame, val_features: pd.DataFrame) -> Tuple:
        """
        Prepare data for training by scaling and converting to TimeSeries
        """
        # Scale target data
        self.scaler_y = Scaler(scaler=MinMaxScaler())
        
        # Convert to TimeSeries
        train_ts = TimeSeries.from_dataframe(train_data, time_col=None)
        val_ts = TimeSeries.from_dataframe(val_data, time_col=None)
        
        # Scale target
        train_y_scaled = self.scaler_y.fit_transform(train_ts).astype(np.float32)
        val_y_scaled = self.scaler_y.transform(val_ts).astype(np.float32)
        
        # Handle features if available
        train_features_scaled = None
        val_features_scaled = None
        
        if train_features is not None and len(train_features.columns) > 0:
            self.scaler_features = Scaler(scaler=MinMaxScaler())
            
            train_features_ts = TimeSeries.from_dataframe(train_features, time_col=None)
            val_features_ts = TimeSeries.from_dataframe(val_features, time_col=None)
            
            train_features_scaled = self.scaler_features.fit_transform(train_features_ts).astype(np.float32)
            val_features_scaled = self.scaler_features.transform(val_features_ts).astype(np.float32)
        
        return train_y_scaled, val_y_scaled, train_features_scaled, val_features_scaled
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
              train_features: Optional[pd.DataFrame] = None, 
              val_features: Optional[pd.DataFrame] = None) -> None:
        """
        Train the TiDE model
        """
        print("Preparing data for training...")
        
        # Prepare data
        train_y, val_y, train_features_scaled, val_features_scaled = self.prepare_data(
            train_data, val_data, train_features, val_features
        )
        
        # Create model if not already created
        if self.model is None:
            feature_dim = len(train_features.columns) if train_features is not None else 0
            self.create_model(target_dim=1, feature_dim=feature_dim)
        
        print("Training TiDE model...")
        print(f"Input chunk length: {self.model.input_chunk_length} steps ({self.model.input_chunk_length * 15} minutes)")
        print(f"Output chunk length: {self.model.output_chunk_length} steps ({self.model.output_chunk_length * 15} minutes)")
        
        if train_features_scaled is not None:
            print(f"Future covariates: {train_features_scaled.values().shape[1]} features")
        
        # Train model
        self.model.fit(
            series=train_y,
            future_covariates=train_features_scaled,
            val_series=val_y,
            val_future_covariates=val_features_scaled,
            verbose=True
        )
        
        # Load best model
        self.model = self.model.load_from_checkpoint(self.model.model_name, best=True)
        self.is_trained = True
        
        print("Training completed!")
    
    def predict(self, test_data: pd.DataFrame, test_features: Optional[pd.DataFrame] = None,
                horizon: int = 192) -> np.ndarray:
        """
        Generate predictions for test data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        print(f"Generating predictions with horizon: {horizon} steps ({horizon * 15} minutes)")
        
        # Prepare test data
        test_ts = TimeSeries.from_dataframe(test_data, time_col=None)
        test_y_scaled = self.scaler_y.transform(test_ts).astype(np.float32)
        
        test_features_scaled = None
        if test_features is not None and len(test_features.columns) > 0:
            test_features_ts = TimeSeries.from_dataframe(test_features, time_col=None)
            test_features_scaled = self.scaler_features.transform(test_features_ts).astype(np.float32)
        
        # Generate forecasts
        forecast_list = self.model.historical_forecasts(
            series=test_y_scaled,
            future_covariates=test_features_scaled,
            start=test_y_scaled.start_time(),
            forecast_horizon=horizon,
            stride=96,  # 24-hour stride
            retrain=False,
            verbose=True,
            last_points_only=False
        )
        
        # Extract last 24 hours of each forecast (if horizon > 96)
        if horizon > 96:
            forecast_24h = [forecast[-96:] for forecast in forecast_list]
        else:
            forecast_24h = forecast_list
        
        # Combine forecasts
        from darts import concatenate
        forecast_combined = concatenate(forecast_24h)
        
        # Inverse transform to original scale
        forecast_unscaled = self.scaler_y.inverse_transform(forecast_combined)
        
        return forecast_unscaled.values().flatten()
    
    def get_actual_values(self, test_data: pd.DataFrame, horizon: int = 192) -> np.ndarray:
        """
        Get actual values aligned with predictions for evaluation
        """
        # This should match the prediction logic
        # For now, return the test data values
        return test_data['demand'].values
    
    def evaluate(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame,
                 train_features: Optional[pd.DataFrame] = None,
                 val_features: Optional[pd.DataFrame] = None,
                 test_features: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Complete evaluation pipeline for TiDE model
        """
        print("=" * 50)
        print("TIDE MODEL EVALUATION")
        print("=" * 50)
        
        # Train model
        self.train(train_data, val_data, train_features, val_features)
        
        # Generate predictions
        predictions = self.predict(test_data, test_features)
        
        # Get actual values
        actual_values = self.get_actual_values(test_data)
        
        # Ensure same length
        min_len = min(len(predictions), len(actual_values))
        predictions = predictions[:min_len]
        actual_values = actual_values[:min_len]
        
        # Calculate metrics
        try:
            from darts.metrics import mape, mae, rmse
            
            # Convert to TimeSeries for Darts metrics
            actual_ts = TimeSeries.from_values(actual_values.reshape(-1, 1))
            pred_ts = TimeSeries.from_values(predictions.reshape(-1, 1))
            
            mape_val = mape(actual_ts, pred_ts)
            mae_val = mae(actual_ts, pred_ts)
            rmse_val = rmse(actual_ts, pred_ts)
        except:
            # Fallback to numpy calculations
            mape_val = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
            mae_val = np.mean(np.abs(actual_values - predictions))
            rmse_val = np.sqrt(np.mean((actual_values - predictions) ** 2))
        
        # Calculate MASE (simplified)
        train_values = train_data['demand'].values
        naive_forecast = np.roll(train_values, -96)[:-96]
        mae_naive = np.mean(np.abs(train_values[96:] - naive_forecast))
        mase_val = mae_val / mae_naive if mae_naive > 0 else float('inf')
        
        results = {
            'mape': mape_val,
            'mae': mae_val,
            'rmse': rmse_val,
            'mase': mase_val,
            'crps': mae_val  # Simplified CRPS for deterministic forecasts
        }
        
        print(f"TiDE Results:")
        print(f"  MAPE: {mape_val:.4f}%")
        print(f"  MAE: {mae_val:.4f}")
        print(f"  RMSE: {rmse_val:.4f}")
        print(f"  MASE: {mase_val:.4f}")
        print(f"  CRPS: {results['crps']:.4f}")
        
        return results


def create_tide_model(model_config: Optional[Dict] = None) -> TiDEModelWrapper:
    """
    Factory function to create TiDE model wrapper
    """
    return TiDEModelWrapper(model_config)


if __name__ == "__main__":
    # Example usage
    model = create_tide_model()
    print("TiDE model created successfully!")
