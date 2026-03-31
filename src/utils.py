"""
Utility functions for the Credit Risk Prediction package.
"""

import os
import pickle
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Any, filepath: str) -> None:
    """Save model to file using joblib."""
    ensure_dir(Path(filepath).parent)
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """Load model from file."""
    return joblib.load(filepath)


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file."""
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """Load CSV file with error handling."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, **kwargs)


def save_csv(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """Save DataFrame to CSV."""
    ensure_dir(Path(filepath).parent)
    df.to_csv(filepath, index=False, **kwargs)


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15
) -> pd.DataFrame:
    """Get feature importance from a model."""
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError(f"Model {type(model)} does not have feature_importances_")
    
    importance = model.feature_importances_
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df.head(top_n)


def get_model_metrics(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Get metrics for multiple models."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        results.append({
            'model': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba)
        })
    
    return pd.DataFrame(results)


def print_section(title: str, width: int = 60) -> None:
    """Print a section header."""
    print(f"\n{'=' * width}")
    print(f" {title}")
    print(f"{'=' * width}\n")


def print_metrics(metrics: Dict[str, float], prefix: str = "  ") -> None:
    """Print metrics in a formatted way."""
    for key, value in metrics.items():
        print(f"{prefix}{key}: {value:.4f}")


def validate_input_data(data: Dict[str, Any]) -> List[str]:
    """Validate input data for prediction."""
    required_fields = [
        'age', 'income', 'credit_score', 'debt_amount',
        'monthly_expenses', 'employment_years', 'loan_amount',
        'existing_credits', 'interest_rate', 'payment_history'
    ]
    
    errors = []
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(data[field], (int, float)):
            errors.append(f"Invalid type for {field}: expected number")
    
    return errors


def format_prediction_response(
    prediction: int,
    probability: float,
    risk_thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """Format prediction response."""
    if probability < risk_thresholds.get('low', 0.3):
        risk_level = 'Low'
    elif probability < risk_thresholds.get('medium', 0.6):
        risk_level = 'Medium'
    else:
        risk_level = 'High'
    
    return {
        'prediction': 'Default' if prediction == 1 else 'No Default',
        'probability': round(float(probability), 4),
        'risk_level': risk_level
    }
