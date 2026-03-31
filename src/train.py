"""
Main training script for Credit Risk Prediction.
Can be run as: python -m src.train or credit-risk-predict
"""

import sys
import os

from .config import get_config
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .logging_config import setup_logging
import pandas as pd
import joblib


def main():
    config = get_config()
    root = config.get_project_root()
    logger = setup_logging()
    
    logger.info("Credit Risk & Loan Default Prediction - Model Training")
    logger.info("=" * 60)
    
    data_path = root / config.data['raw_dir'] / config.data['dataset_file']
    model_path = root / config.model['output_dir'] / config.model['model_file']
    
    logger.info("[1/6] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y, feature_cols, scaler = preprocessor.preprocess(str(data_path))
    logger.info(f"  - Features shape: {X.shape}")
    logger.info(f"  - Target distribution: {dict(y.value_counts())}")
    
    logger.info("[2/6] Engineering features...")
    engineer = FeatureEngineer()
    df = pd.DataFrame(X, columns=feature_cols)
    df['default'] = y
    df = engineer.create_all_features(df)
    
    df = df.dropna()
    logger.info(f"  - New features created: {engineer.new_features}")
    
    X = df.drop('default', axis=1)
    y = df['default']
    logger.info(f"  - Total features: {len(X.columns)}")
    
    logger.info("[3/6] Splitting data...")
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    logger.info(f"  - Training set: {X_train.shape[0]} samples")
    logger.info(f"  - Test set: {X_test.shape[0]} samples")
    
    logger.info("[4/6] Training models...")
    trainer.train_models(X_train, y_train)
    
    logger.info("[5/6] Cross-validation...")
    trainer.cross_validate_models(X_train, y_train)
    
    logger.info("[6/6] Evaluating models...")
    results = trainer.evaluate_models(X_test, y_test)
    
    best_model, best_name = trainer.select_best_model()
    
    trainer.save_best_model(str(model_path))
    
    processed_dir = root / config.data['processed_dir']
    preprocessor.save_preprocessor(str(processed_dir / config.model['preprocessor_file']))
    joblib.dump(scaler, str(processed_dir / config.model['scaler_file']))
    
    df_processed = pd.DataFrame(X, columns=X.columns)
    df_processed['default'] = y
    df_processed.to_csv(str(processed_dir / config.data['processed_file']), index=False)
    
    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
