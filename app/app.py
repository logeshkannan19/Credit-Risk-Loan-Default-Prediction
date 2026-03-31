"""
Flask API for Credit Risk Prediction.
Can be run as: python -m src.app or python app/app.py
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import get_config
from feature_engineering import FeatureEngineer
from logging_config import setup_logging


app = Flask(__name__)

config = get_config()
logger = setup_logging()

model = None
feature_columns = None


def load_model_and_features():
    """Load model and feature columns."""
    global model, feature_columns
    
    root = config.get_project_root()
    model_path = root / config.model['output_dir'] / config.model['model_file']
    processed_path = root / config.data['processed_dir'] / config.data['processed_file']
    
    try:
        model = joblib.load(str(model_path))
        
        df = pd.read_csv(str(processed_path))
        feature_columns = [col for col in df.columns if col != 'default']
        
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Features: {feature_columns}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Credit Risk Prediction API is running'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict loan default risk.
    
    Request body (JSON):
    {
        "age": 35,
        "income": 75000,
        "credit_score": 650,
        "debt_amount": 15000,
        "monthly_expenses": 2500,
        "employment_years": 5,
        "loan_amount": 25000,
        "existing_credits": 2,
        "interest_rate": 12.5,
        "payment_history": 85
    }
    
    Response (JSON):
    {
        "prediction": "No Default" | "Default",
        "probability": 0.XX,
        "risk_level": "Low" | "Medium" | "High"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        required_fields = [
            'age', 'income', 'credit_score', 'debt_amount',
            'monthly_expenses', 'employment_years', 'loan_amount',
            'existing_credits', 'interest_rate', 'payment_history'
        ]
        
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields
            }), 400
        
        input_df = pd.DataFrame([data])
        
        engineer = FeatureEngineer()
        input_df = engineer.create_all_features(input_df)
        
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        X = input_df[feature_columns]
        X = X.fillna(0)
        
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]
        
        risk = config.risk
        if probability < risk.get('low', 0.3):
            risk_level = 'Low'
        elif probability < risk.get('medium', 0.6):
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        result = {
            'prediction': 'Default' if prediction == 1 else 'No Default',
            'probability': round(float(probability), 4),
            'risk_level': risk_level
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch predict loan default risk.
    
    Request body (JSON):
    {
        "applicants": [
            {"age": 35, "income": 75000, ...},
            {"age": 45, "income": 60000, ...}
        ]
    }
    
    Response (JSON):
    {
        "predictions": [
            {"prediction": "No Default", "probability": 0.XX, "risk_level": "Low"},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'applicants' not in data:
            return jsonify({'error': 'No applicants data provided'}), 400
        
        applicants = data['applicants']
        predictions = []
        risk = config.risk
        
        for applicant in applicants:
            try:
                input_df = pd.DataFrame([applicant])
                
                engineer = FeatureEngineer()
                input_df = engineer.create_all_features(input_df)
                
                for col in feature_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                X = input_df[feature_columns]
                X = X.fillna(0)
                
                prediction = model.predict(X)[0]
                probability = model.predict_proba(X)[0, 1]
                
                if probability < risk.get('low', 0.3):
                    risk_level = 'Low'
                elif probability < risk.get('medium', 0.6):
                    risk_level = 'Medium'
                else:
                    risk_level = 'High'
                
                predictions.append({
                    'prediction': 'Default' if prediction == 1 else 'No Default',
                    'probability': round(float(probability), 4),
                    'risk_level': risk_level
                })
                
            except Exception as e:
                predictions.append({
                    'error': str(e),
                    'prediction': 'Error',
                    'probability': 0,
                    'risk_level': 'Unknown'
                })
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information."""
    return jsonify({
        'model_type': type(model).__name__,
        'features': feature_columns,
        'n_features': len(feature_columns)
    })


if __name__ == '__main__':
    load_model_and_features()
    
    api_config = config.api
    app.run(
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 5000),
        debug=api_config.get('debug', True)
    )
