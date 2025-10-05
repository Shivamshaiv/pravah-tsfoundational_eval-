"""
Models package for electricity demand forecasting evaluation
"""

from .tide_model import TiDEModelWrapper, create_tide_model
from .baseline_models import (
    NaiveModel, SeasonalNaiveModel, MovingAverageModel,
    evaluate_baseline_model, create_baseline_models
)
from .darts_models import DartsModelWrapper, create_darts_models
from .foundation_models import FoundationModelWrapper, create_foundation_models

__all__ = [
    'TiDEModelWrapper', 'create_tide_model',
    'NaiveModel', 'SeasonalNaiveModel', 'MovingAverageModel',
    'evaluate_baseline_model', 'create_baseline_models',
    'DartsModelWrapper', 'create_darts_models',
    'FoundationModelWrapper', 'create_foundation_models'
]
