import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
import pandas as pd
import joblib


def main():
    print("=" * 60)
    print("Credit Risk & Loan Default Prediction - Model Training")
    print("=" * 60)
    
    data_path = '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/raw/credit_data.csv'
    model_path = '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/models/model.pkl'
    
    print("\n[1/6] Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y, feature_cols, scaler = preprocessor.preprocess(data_path)
    print(f"  - Features shape: {X.shape}")
    print(f"  - Target distribution: {dict(y.value_counts())}")
    
    print("\n[2/6] Engineering features...")
    engineer = FeatureEngineer()
    df = pd.DataFrame(X, columns=feature_cols)
    df['default'] = y
    df = engineer.create_all_features(df)
    
    df = df.dropna()
    print(f"  - New features created: {engineer.new_features}")
    
    X = df.drop('default', axis=1)
    y = df['default']
    print(f"  - Total features: {len(X.columns)}")
    
    print("\n[3/6] Splitting data...")
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Test set: {X_test.shape[0]} samples")
    
    print("\n[4/6] Training models...")
    trainer.train_models(X_train, y_train)
    
    print("\n[5/6] Cross-validation...")
    trainer.cross_validate_models(X_train, y_train)
    
    print("\n[6/6] Evaluating models...")
    results = trainer.evaluate_models(X_test, y_test)
    
    best_model, best_name = trainer.select_best_model()
    
    trainer.save_best_model(model_path)
    
    preprocessor.save_preprocessor('/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/processed/preprocessor.pkl')
    joblib.dump(scaler, '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/processed/scaler.pkl')
    
    df_processed = pd.DataFrame(X, columns=X.columns)
    df_processed['default'] = y
    df_processed.to_csv('/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/processed/credit_data_processed.csv', index=False)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
