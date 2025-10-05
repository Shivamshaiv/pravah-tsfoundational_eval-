#!/usr/bin/env python3
"""
Darts-based Deep Learning Models for Electricity Demand Forecasting
- PatchTST
- TFT (Temporal Fusion Transformer)
- N-BEATS
- DLinear
- NHiTS
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks import EarlyStopping

# Import GPU utilities
from models.gpu_utils import get_gpu_config, is_gpu_available, get_trainer_kwargs, optimize_batch_size


class DartsModelWrapper:
    """
    Wrapper for Darts deep learning models
    """
    
    def __init__(self, model_name: str, model_config: Optional[Dict] = None):
        self.model_name = model_name
        self.model_config = model_config or self._get_default_config(model_name)
        self.model = None
        self.scaler_y = None
        self.scaler_features = None
        self.is_trained = False
        self.gpu_config = get_gpu_config()
        
    def _get_default_config(self, model_name: str) -> Dict:
        """Get default configuration for each model"""
        
        base_config = {
            'n_epochs': 50,
            'batch_size': 32,
            'random_state': 42,
            'save_checkpoints': True,
            'force_reset': True
        }
        
        if model_name == "PatchTST":
            return {
                **base_config,
                'input_chunk_length': 192,
                'output_chunk_length': 192,
                'patch_len': 16,
                'stride': 8,
                'd_model': 128,
                'nhead': 8,
                'num_encoder_layers': 3,
                'dropout': 0.2,
                'learning_rate': 1e-3
            }
        
        elif model_name == "TFT":
            return {
                **base_config,
                'input_chunk_length': 192,
                'output_chunk_length': 192,
                'hidden_size': 128,
                'lstm_layers': 2,
                'num_attention_heads': 4,
                'dropout': 0.1,
                'learning_rate': 1e-3
            }
        
        elif model_name == "NBEATS":
            return {
                **base_config,
                'input_chunk_length': 192,
                'output_chunk_length': 192,
                'num_stacks': 30,
                'num_blocks': 1,
                'num_layers': 4,
                'layer_widths': 256,
                'expansion_coefficient_dim': 5,
                'learning_rate': 1e-3
            }
        
        elif model_name == "DLinear":
            return {
                **base_config,
                'input_chunk_length': 192,
                'output_chunk_length': 192,
                'kernel_size': 25,
                'learning_rate': 1e-3
            }
        
        elif model_name == "NHiTS":
            return {
                **base_config,
                'input_chunk_length': 192,
                'output_chunk_length': 192,
                'num_stacks': 3,
                'num_blocks': 1,
                'num_layers': 2,
                'layer_widths': 512,
                'learning_rate': 1e-3
            }
        
        else:
            return base_config
    
    def create_model(self):
        """Create model instance"""
        config = self.model_config
        
        # Get optimized batch size for device
        batch_size = optimize_batch_size(config.get('batch_size', 32))
        config['batch_size'] = batch_size
        
        print(f"Creating {self.model_name} model")
        print(f"Device: {self.gpu_config.device} ({self.gpu_config.device_name})")
        print(f"Batch size: {batch_size}")
        
        # Common trainer kwargs optimized for device
        trainer_kwargs = get_trainer_kwargs(
            gradient_clip_val=1.0,
            precision="32-true",
            callbacks=[EarlyStopping(monitor="val_loss", patience=10)]
        )
        
        try:
            if self.model_name == "PatchTST":
                from darts.models import PatchTSTModel
                self.model = PatchTSTModel(
                    input_chunk_length=config['input_chunk_length'],
                    output_chunk_length=config['output_chunk_length'],
                    patch_len=config['patch_len'],
                    stride=config['stride'],
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_encoder_layers=config['num_encoder_layers'],
                    dropout=config['dropout'],
                    n_epochs=config['n_epochs'],
                    batch_size=config['batch_size'],
                    optimizer_kwargs={"lr": config['learning_rate']},
                    pl_trainer_kwargs=trainer_kwargs,
                    save_checkpoints=config['save_checkpoints'],
                    force_reset=config['force_reset'],
                    random_state=config['random_state']
                )
            
            elif self.model_name == "TFT":
                from darts.models import TFTModel
                self.model = TFTModel(
                    input_chunk_length=config['input_chunk_length'],
                    output_chunk_length=config['output_chunk_length'],
                    hidden_size=config['hidden_size'],
                    lstm_layers=config['lstm_layers'],
                    num_attention_heads=config['num_attention_heads'],
                    dropout=config['dropout'],
                    n_epochs=config['n_epochs'],
                    batch_size=config['batch_size'],
                    optimizer_kwargs={"lr": config['learning_rate']},
                    pl_trainer_kwargs=trainer_kwargs,
                    save_checkpoints=config['save_checkpoints'],
                    force_reset=config['force_reset'],
                    random_state=config['random_state']
                )
            
            elif self.model_name == "NBEATS":
                from darts.models import NBEATSModel
                self.model = NBEATSModel(
                    input_chunk_length=config['input_chunk_length'],
                    output_chunk_length=config['output_chunk_length'],
                    num_stacks=config['num_stacks'],
                    num_blocks=config['num_blocks'],
                    num_layers=config['num_layers'],
                    layer_widths=config['layer_widths'],
                    expansion_coefficient_dim=config['expansion_coefficient_dim'],
                    n_epochs=config['n_epochs'],
                    batch_size=config['batch_size'],
                    optimizer_kwargs={"lr": config['learning_rate']},
                    pl_trainer_kwargs=trainer_kwargs,
                    save_checkpoints=config['save_checkpoints'],
                    force_reset=config['force_reset'],
                    random_state=config['random_state']
                )
            
            elif self.model_name == "DLinear":
                from darts.models import DLinearModel
                self.model = DLinearModel(
                    input_chunk_length=config['input_chunk_length'],
                    output_chunk_length=config['output_chunk_length'],
                    kernel_size=config['kernel_size'],
                    n_epochs=config['n_epochs'],
                    batch_size=config['batch_size'],
                    optimizer_kwargs={"lr": config['learning_rate']},
                    pl_trainer_kwargs=trainer_kwargs,
                    save_checkpoints=config['save_checkpoints'],
                    force_reset=config['force_reset'],
                    random_state=config['random_state']
                )
            
            elif self.model_name == "NHiTS":
                from darts.models import NHiTSModel
                self.model = NHiTSModel(
                    input_chunk_length=config['input_chunk_length'],
                    output_chunk_length=config['output_chunk_length'],
                    num_stacks=config['num_stacks'],
                    num_blocks=config['num_blocks'],
                    num_layers=config['num_layers'],
                    layer_widths=config['layer_widths'],
                    n_epochs=config['n_epochs'],
                    batch_size=config['batch_size'],
                    optimizer_kwargs={"lr": config['learning_rate']},
                    pl_trainer_kwargs=trainer_kwargs,
                    save_checkpoints=config['save_checkpoints'],
                    force_reset=config['force_reset'],
                    random_state=config['random_state']
                )
            
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
                
        except ImportError as e:
            print(f"Error importing {self.model_name}: {e}")
            raise
        
        return self.model
    
    def prepare_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                    train_features: Optional[pd.DataFrame] = None,
                    val_features: Optional[pd.DataFrame] = None):
        """Prepare and scale data"""
        
        # Scale target
        self.scaler_y = Scaler(scaler=MinMaxScaler())
        
        train_ts = TimeSeries.from_dataframe(train_data, time_col=None)
        val_ts = TimeSeries.from_dataframe(val_data, time_col=None)
        
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
              val_features: Optional[pd.DataFrame] = None):
        """Train the model"""
        
        print(f"Training {self.model_name} model...")
        
        # Prepare data
        train_y, val_y, train_features_scaled, val_features_scaled = self.prepare_data(
            train_data, val_data, train_features, val_features
        )
        
        # Create model if not already created
        if self.model is None:
            self.create_model()
        
        # Train
        self.model.fit(
            series=train_y,
            future_covariates=train_features_scaled,
            val_series=val_y,
            val_future_covariates=val_features_scaled,
            verbose=False
        )
        
        self.is_trained = True
        print(f"{self.model_name} training completed!")
    
    def predict(self, test_data: pd.DataFrame, test_features: Optional[pd.DataFrame] = None,
                horizon: int = 192) -> np.ndarray:
        """Generate predictions"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
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
            stride=96,
            retrain=False,
            verbose=False,
            last_points_only=False
        )
        
        # Extract last 24 hours
        from darts import concatenate
        forecast_24h = [forecast[-96:] for forecast in forecast_list]
        forecast_combined = concatenate(forecast_24h)
        forecast_unscaled = self.scaler_y.inverse_transform(forecast_combined)
        
        return forecast_unscaled.values().flatten()
    
    def evaluate(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame,
                 train_features: Optional[pd.DataFrame] = None,
                 val_features: Optional[pd.DataFrame] = None,
                 test_features: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Complete evaluation pipeline"""
        
        print(f"=" * 50)
        print(f"{self.model_name} MODEL EVALUATION")
        print("=" * 50)
        
        # Train
        self.train(train_data, val_data, train_features, val_features)
        
        # Predict
        predictions = self.predict(test_data, test_features)
        actual_values = test_data['demand'].values
        
        # Align lengths
        min_len = min(len(predictions), len(actual_values))
        predictions = predictions[:min_len]
        actual_values = actual_values[:min_len]
        
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


def create_darts_models():
    """Factory function to create all Darts models"""
    return {
        'PatchTST': DartsModelWrapper('PatchTST'),
        'TFT': DartsModelWrapper('TFT'),
        'NBEATS': DartsModelWrapper('NBEATS'),
        'DLinear': DartsModelWrapper('DLinear'),
        'NHiTS': DartsModelWrapper('NHiTS')
    }


if __name__ == "__main__":
    models = create_darts_models()
    print("Darts models created:")
    for name in models.keys():
        print(f"  - {name}")
