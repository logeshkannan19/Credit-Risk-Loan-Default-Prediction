"""
Credit Risk & Loan Default Prediction Package

A complete machine learning project for credit risk assessment and loan default prediction.
"""

__version__ = "1.0.0"
__author__ = "Logesh Kannan"
__license__ = "MIT"

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator
from .config import Config, get_config
from .utils import (
    get_project_root,
    ensure_dir,
    save_model,
    load_model,
    validate_input_data,
    format_prediction_response,
)

__all__ = [
    "DataPreprocessor",
    "FeatureEngineer",
    "ModelTrainer",
    "ModelEvaluator",
    "Config",
    "get_config",
    "get_project_root",
    "ensure_dir",
    "save_model",
    "load_model",
    "validate_input_data",
    "format_prediction_response",
]
