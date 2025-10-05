#!/usr/bin/env python3
"""
GIFTEval Model Evaluation Framework for Electricity Demand Forecasting

This script evaluates multiple time series forecasting models from the GIFTEval leaderboard
on electricity demand data with proper train/validation/test splits.

Data splits:
- Train: 2016-2022
- Validation: 2023
- Test: 2024

Evaluation metrics: MAPE, MASE, CRPS
Forecast horizon: 48 hours with 24-hour stride
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pytz
from pathlib import Path

# Time series libraries
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae, rmse
from sklearn.preprocessing import MinMaxScaler

# Weather data
from meteostat import Hourly
import holidays

# Model imports (will be added as we implement them)
import torch
import torch.nn as nn

# GPU utilities
from models.gpu_utils import print_gpu_info

warnings.filterwarnings("ignore")

class ElectricityDemandEvaluator:
    """
    Main evaluation framework for electricity demand forecasting models
    """
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.results_path = Path("results")
        self.results_path.mkdir(exist_ok=True)
        
        # Time zones
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Weather stations near Delhi
        self.station_ids = ['42182']  # Add more stations as needed
        
        # Evaluation parameters
        self.forecast_horizon_hours = 48
        self.stride_hours = 24
        self.forecast_horizon_15min = self.forecast_horizon_hours * 4  # 192 steps for 15-min data
        self.stride_15min = self.stride_hours * 4  # 96 steps for 15-min data
        
        # Data splits
        self.train_start = "2016-01-01"
        self.train_end = "2022-12-31"
        self.val_start = "2023-01-01"
        self.val_end = "2023-12-31"
        self.test_start = "2024-01-01"
        self.test_end = "2024-12-31"
        
        # Initialize data
        self.demand_data = None
        self.weather_data = None
        self.calendar_features = None
        
    def load_demand_data(self, filename: str = "merged_timeseries.csv") -> pd.DataFrame:
        """
        Load electricity demand data
        
        Expected format:
        - datetime column with timestamps
        - demand column with electricity demand values
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            # Create sample data for testing if file doesn't exist
            print(f"Warning: {file_path} not found. Creating sample data for testing.")
            return self._create_sample_demand_data()
        
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Handle datetime column
        datetime_col = None
        for col in ['datetime', 'ds', 'timestamp', 'time']:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col is None:
            raise ValueError("No datetime column found. Expected one of: datetime, ds, timestamp, time")
        
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df.set_index(datetime_col, inplace=True)
        df = df.sort_index()
        
        # Handle demand column
        demand_col = None
        for col in ['demand', 'y', 'value', 'electricity_demand']:
            if col in df.columns:
                demand_col = col
                break
        
        if demand_col is None:
            raise ValueError("No demand column found. Expected one of: demand, y, value, electricity_demand")
        
        # Rename to standard column name
        if demand_col != 'demand':
            df['demand'] = df[demand_col]
            df = df.drop(columns=[demand_col])
        
        # Ensure timezone-naive (assuming IST)
        if df.index.tz is not None:
            print(f"Converting demand data from {df.index.tz} to timezone-naive IST")
            df.index = df.index.tz_convert('Asia/Kolkata').tz_localize(None)
        
        self.demand_data = df
        print(f"Loaded demand data: {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
    
    def _create_sample_demand_data(self) -> pd.DataFrame:
        """Create sample electricity demand data for testing"""
        print("Creating sample electricity demand data...")
        
        # Generate timestamps every 15 minutes from 2016 to 2024
        start_date = datetime(2016, 1, 1)
        end_date = datetime(2024, 12, 31, 23, 45)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='15T')
        
        # Generate realistic electricity demand pattern
        np.random.seed(42)
        base_demand = 1000  # MW
        
        # Seasonal pattern
        seasonal = 200 * np.sin(2 * np.pi * timestamps.dayofyear / 365.25)
        
        # Weekly pattern
        weekly = 100 * np.sin(2 * np.pi * timestamps.dayofweek / 7)
        
        # Daily pattern
        daily = 300 * np.sin(2 * np.pi * timestamps.hour / 24)
        
        # Random noise
        noise = np.random.normal(0, 50, len(timestamps))
        
        demand = base_demand + seasonal + weekly + daily + noise
        
        df = pd.DataFrame({
            'demand': demand
        }, index=timestamps)
        
        # Save sample data
        self.data_path.mkdir(exist_ok=True)
        df.to_csv(self.data_path / "sample_demand_timeseries.csv")
        print(f"Sample data saved to {self.data_path / 'sample_demand_timeseries.csv'}")
        
        # Set the demand data
        self.demand_data = df
        print(f"Sample demand data created: {len(df)} records from {df.index.min()} to {df.index.max()}")
        
        return df
    
    def load_weather_data(self) -> pd.DataFrame:
        """
        Load weather data from Meteostat
        """
        if self.demand_data is None:
            print("Demand data not loaded, loading sample data first...")
            self.load_demand_data()
        
        print("Loading weather data from Meteostat...")
        
        start = self.demand_data.index.min()
        end = self.demand_data.index.max()
        weather_data = {}
        
        for station_id in self.station_ids:
            try:
                print(f"Fetching weather data for station: {station_id}")
                
                # Convert IST times to UTC for the API call
                start_timestamp = pd.Timestamp(start)
                end_timestamp = pd.Timestamp(end)
                
                # Since the demand data is timezone-naive (assumed IST), localize it first
                start_ist_aware = self.ist.localize(start_timestamp)
                end_ist_aware = self.ist.localize(end_timestamp)
                
                # Convert to UTC
                start_utc_aware = start_ist_aware.astimezone(pytz.UTC)
                end_utc_aware = end_ist_aware.astimezone(pytz.UTC)
                
                # Make timezone-naive for Meteostat API
                start_utc = start_utc_aware.tz_localize(None)
                end_utc = end_utc_aware.tz_localize(None)
                
                data = Hourly(station_id, start_utc, end_utc).fetch()
                
                if data.empty:
                    print(f"No data returned for station {station_id}")
                    continue
                
                # Select relevant weather variables
                data = data[["temp", "dwpt", "rhum", "wspd", "pres"]]
                
                # Convert UTC back to IST and make timezone-naive
                data.index = data.index.tz_convert('Asia/Kolkata').tz_localize(None)
                
                # Resample to 15-minute frequency using interpolation
                data_15min = data.resample('15T').interpolate(method='linear')
                
                weather_data[station_id] = data_15min
                print(f"Loaded weather data for station {station_id}: {len(data_15min)} records")
                
            except Exception as e:
                print(f"Error loading weather data for station {station_id}: {e}")
                continue
        
        if not weather_data:
            print("Warning: No weather data loaded. Creating sample weather data.")
            return self._create_sample_weather_data()
        
        # Combine weather data from multiple stations (if available)
        if len(weather_data) == 1:
            self.weather_data = list(weather_data.values())[0]
        else:
            # Average across stations
            combined_weather = pd.concat(weather_data.values(), axis=1)
            self.weather_data = combined_weather.groupby(level=0, axis=1).mean()
        
        return self.weather_data
    
    def _create_sample_weather_data(self) -> pd.DataFrame:
        """Create sample weather data for testing"""
        print("Creating sample weather data...")
        
        if self.demand_data is None:
            raise ValueError("Load demand data first")
        
        timestamps = self.demand_data.index
        np.random.seed(42)
        
        # Generate realistic weather patterns
        weather_data = pd.DataFrame(index=timestamps)
        
        # Temperature (Celsius) - seasonal pattern
        temp_base = 25
        temp_seasonal = 15 * np.sin(2 * np.pi * timestamps.dayofyear / 365.25)
        temp_noise = np.random.normal(0, 3, len(timestamps))
        weather_data['temp'] = temp_base + temp_seasonal + temp_noise
        
        # Dew point (slightly lower than temperature)
        weather_data['dwpt'] = weather_data['temp'] - np.random.uniform(2, 8, len(timestamps))
        
        # Relative humidity (%)
        weather_data['rhum'] = np.clip(50 + 30 * np.sin(2 * np.pi * timestamps.hour / 24) + 
                                      np.random.normal(0, 10, len(timestamps)), 0, 100)
        
        # Wind speed (m/s)
        weather_data['wspd'] = np.clip(np.random.gamma(2, 1.5, len(timestamps)), 0, 20)
        
        # Pressure (hPa)
        weather_data['pres'] = 1013 + np.random.normal(0, 5, len(timestamps))
        
        self.weather_data = weather_data
        print(f"Sample weather data created: {len(weather_data)} records")
        return weather_data
    
    def create_calendar_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Create calendar features (hour, day of week, month, holidays, etc.)
        """
        features = pd.DataFrame(index=index)
        
        # Cyclical encoding for time features
        features["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
        features["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
        features["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
        features["month_sin"] = np.sin(2 * np.pi * index.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * index.month / 12)
        
        # Binary features
        features["is_weekend"] = (index.dayofweek >= 5).astype(int)
        
        # 15-minute specific features
        features["quarter_hour_sin"] = np.sin(2 * np.pi * index.minute / 60)
        features["quarter_hour_cos"] = np.cos(2 * np.pi * index.minute / 60)
        
        # Indian holidays
        in_holidays = holidays.country_holidays("IN", years=range(index.year.min(), index.year.max() + 1))
        features["is_holiday"] = index.normalize().isin(pd.to_datetime(list(in_holidays.keys()))).astype(int)
        
        self.calendar_features = features
        return features
    
    def prepare_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare train/validation/test data splits
        """
        if self.demand_data is None:
            print("Demand data not loaded, loading sample data first...")
            self.load_demand_data()
        
        # Create calendar features
        self.create_calendar_features(self.demand_data.index)
        
        # Load weather data if not already loaded
        if self.weather_data is None:
            self.load_weather_data()
        
        # Align all data to common time index
        common_index = self.demand_data.index
        
        # Prepare features DataFrame
        features_df = pd.DataFrame(index=common_index)
        
        # Add calendar features
        features_df = features_df.join(self.calendar_features)
        
        # Add weather features
        if self.weather_data is not None:
            # Align weather data to common index
            weather_aligned = self.weather_data.reindex(common_index).interpolate(method='linear')
            features_df = features_df.join(weather_aligned)
        
        # Create target DataFrame
        target_df = self.demand_data[['demand']].copy()
        
        # Split data
        train_data = target_df[self.train_start:self.train_end]
        val_data = target_df[self.val_start:self.val_end]
        test_data = target_df[self.test_start:self.test_end]
        
        train_features = features_df[self.train_start:self.train_end]
        val_features = features_df[self.val_start:self.val_end]
        test_features = features_df[self.test_start:self.test_end]
        
        print(f"Data splits prepared:")
        print(f"Train: {len(train_data)} records ({train_data.index.min()} to {train_data.index.max()})")
        print(f"Validation: {len(val_data)} records ({val_data.index.min()} to {val_data.index.max()})")
        print(f"Test: {len(test_data)} records ({test_data.index.min()} to {test_data.index.max()})")
        print(f"Features: {features_df.shape[1]} features")
        
        return (train_data, val_data, test_data), (train_features, val_features, test_features)
    
    def calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        """
        Calculate Mean Absolute Scaled Error (MASE)
        """
        mae_forecast = np.mean(np.abs(y_true - y_pred))
        
        # Calculate naive forecast error (seasonal naive)
        naive_forecast = np.roll(y_train, -96)  # 96 = 24 hours * 4 (15-min intervals)
        naive_forecast = naive_forecast[:-96]  # Remove last 96 points
        y_train_trimmed = y_train[96:]  # Remove first 96 points
        mae_naive = np.mean(np.abs(y_train_trimmed - naive_forecast))
        
        if mae_naive == 0:
            return float('inf')
        
        return mae_forecast / mae_naive
    
    def calculate_crps(self, y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_std: np.ndarray) -> float:
        """
        Calculate Continuous Ranked Probability Score (CRPS) for normal distribution
        """
        # For deterministic forecasts, CRPS reduces to MAE
        # For probabilistic forecasts, we would need the full distribution
        return np.mean(np.abs(y_true - y_pred_mean))
    
    def evaluate_forecasts(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_train: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Calculate evaluation metrics for forecasts
        """
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {f"{model_name}_mape": float('inf'),
                   f"{model_name}_mase": float('inf'),
                   f"{model_name}_crps": float('inf')}
        
        # Calculate metrics
        mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mase_val = self.calculate_mase(y_true, y_pred, y_train)
        crps_val = self.calculate_crps(y_true, y_pred, np.zeros_like(y_pred))  # Deterministic
        
        return {
            f"{model_name}_mape": mape_val,
            f"{model_name}_mase": mase_val,
            f"{model_name}_crps": crps_val
        }
    
    def run_evaluation(self, models: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Run evaluation for multiple models
        """
        print("=" * 80)
        print("ELECTRICITY DEMAND FORECASTING EVALUATION")
        print("=" * 80)
        print()
        
        # Print GPU information
        print_gpu_info()
        print()
        
        # Prepare data
        (train_data, val_data, test_data), (train_features, val_features, test_features) = self.prepare_data_splits()
        
        results = {}
        
        for model_name in models:
            print(f"\nEvaluating model: {model_name}")
            print("-" * 40)
            
            try:
                # This is where we would implement each model
                # For now, we'll create a placeholder
                model_results = self._evaluate_model_placeholder(
                    model_name, train_data, val_data, test_data, 
                    train_features, val_features, test_features
                )
                results[model_name] = model_results
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {
                    f"{model_name}_mape": float('inf'),
                    f"{model_name}_mase": float('inf'),
                    f"{model_name}_crps": float('inf')
                }
        
        # Save results
        self._save_results(results)
        return results
    
    def _evaluate_model_placeholder(self, model_name: str, train_data: pd.DataFrame,
                                   val_data: pd.DataFrame, test_data: pd.DataFrame,
                                   train_features: pd.DataFrame, val_features: pd.DataFrame,
                                   test_features: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate specific models
        """
        # Baseline models
        if model_name in ["Naive", "SeasonalNaive", "MovingAverage"]:
            return self._evaluate_baseline_model(model_name, train_data, val_data, test_data)
        
        # Deep learning models
        elif model_name == "TiDE":
            return self._evaluate_tide_model(train_data, val_data, test_data, 
                                           train_features, val_features, test_features)
        
        # Darts deep learning models
        elif model_name in ["PatchTST", "TFT", "NBEATS", "DLinear", "NHiTS"]:
            return self._evaluate_darts_model(model_name, train_data, val_data, test_data,
                                             train_features, val_features, test_features)
        
        # Foundation models
        elif model_name in ["TimesFM", "Moirai", "Chronos"]:
            return self._evaluate_foundation_model(model_name, train_data, val_data, test_data,
                                                  train_features, val_features, test_features)
        
        # Not implemented
        else:
            print(f"Model {model_name} not yet implemented. Skipping...")
            return {
                "mape": float('nan'),
                "mase": float('nan'),
                "crps": float('nan'),
                "mae": float('nan'),
                "rmse": float('nan')
            }
    
    def _evaluate_baseline_model(self, model_name: str, train_data: pd.DataFrame,
                                val_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate baseline models (Naive, SeasonalNaive, MovingAverage)
        """
        try:
            from models.baseline_models import create_baseline_models, evaluate_baseline_model
            
            # Create baseline models
            baseline_models = create_baseline_models()
            
            if model_name not in baseline_models:
                raise ValueError(f"Unknown baseline model: {model_name}")
            
            # Get the specific model
            model = baseline_models[model_name]
            
            # Evaluate on test data
            results = evaluate_baseline_model(model, train_data, test_data)
            
            return results
            
        except Exception as e:
            print(f"Error evaluating {model_name} model: {e}")
            import traceback
            traceback.print_exc()
            return {
                "mape": float('inf'),
                "mase": float('inf'),
                "crps": float('inf')
            }
    
    def _evaluate_tide_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                           test_data: pd.DataFrame, train_features: pd.DataFrame, 
                           val_features: pd.DataFrame, test_features: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate TiDE model
        """
        try:
            from models.tide_model import create_tide_model
            
            # Create and train model
            model = create_tide_model()
            results = model.evaluate(
                train_data, val_data, test_data,
                train_features, val_features, test_features
            )
            
            return results
            
        except Exception as e:
            print(f"Error evaluating TiDE model: {e}")
            import traceback
            traceback.print_exc()
            return {
                "mape": float('inf'),
                "mase": float('inf'),
                "crps": float('inf')
            }
    
    def _evaluate_darts_model(self, model_name: str, train_data: pd.DataFrame, 
                             val_data: pd.DataFrame, test_data: pd.DataFrame,
                             train_features: pd.DataFrame, val_features: pd.DataFrame, 
                             test_features: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate Darts deep learning models (PatchTST, TFT, NBEATS, DLinear, NHiTS)
        """
        try:
            from models.darts_models import DartsModelWrapper
            
            # Create and train model
            model = DartsModelWrapper(model_name)
            results = model.evaluate(
                train_data, val_data, test_data,
                train_features, val_features, test_features
            )
            
            return results
            
        except Exception as e:
            print(f"Error evaluating {model_name} model: {e}")
            import traceback
            traceback.print_exc()
            return {
                "mape": float('inf'),
                "mae": float('inf'),
                "rmse": float('inf'),
                "mase": float('inf'),
                "crps": float('inf')
            }
    
    def _evaluate_foundation_model(self, model_name: str, train_data: pd.DataFrame, 
                                  val_data: pd.DataFrame, test_data: pd.DataFrame,
                                  train_features: pd.DataFrame, val_features: pd.DataFrame, 
                                  test_features: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate foundation models (TimesFM, Moirai, Chronos)
        """
        try:
            from models.foundation_models import FoundationModelWrapper
            
            # Create and evaluate model
            model = FoundationModelWrapper(model_name)
            results = model.evaluate(
                train_data, val_data, test_data,
                train_features, val_features, test_features
            )
            
            return results
            
        except Exception as e:
            print(f"Error evaluating {model_name} model: {e}")
            import traceback
            traceback.print_exc()
            return {
                "mape": float('inf'),
                "mae": float('inf'),
                "rmse": float('inf'),
                "mase": float('inf'),
                "crps": float('inf')
            }
    
    def _save_results(self, results: Dict[str, Dict[str, float]]):
        """
        Save evaluation results to file
        """
        # Create results DataFrame
        results_df = pd.DataFrame(results).T
        
        # Save to CSV
        results_file = self.results_path / "evaluation_results.csv"
        results_df.to_csv(results_file)
        
        # Save to JSON
        results_json = self.results_path / "evaluation_results.json"
        with open(results_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"- {results_file}")
        print(f"- {results_json}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(results_df.round(4))


def main():
    """
    Main evaluation function
    """
    print("=" * 80)
    print("GIFTEval Model Evaluation Framework")
    print("=" * 80)
    print()
    
    # Print GPU info at startup
    print_gpu_info()
    
    # Initialize evaluator
    evaluator = ElectricityDemandEvaluator()
    
    # Load data (will use actual data from data/merged_timeseries.csv)
    evaluator.load_demand_data()
    
    # Define models to evaluate
    models_to_evaluate = [
        # Baseline models (fast)
        "Naive",
        "SeasonalNaive",
        "MovingAverage",
        
        # Deep learning models from Darts
        "TiDE",
        "PatchTST",
        "TFT", 
        "NBEATS",
        "DLinear",
        "NHiTS",
        
        # Foundation models (zero-shot)
        "TimesFM",
        "Moirai",
        "Chronos",
    ]
    
    # Run evaluation
    results = evaluator.run_evaluation(models_to_evaluate)
    
    return results


if __name__ == "__main__":
    results = main()
