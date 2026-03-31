import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets.
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Target
        test_size : float
            Test set proportion
        random_state : int
            Random seed
        
        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """Initialize all models to train."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1
            )
        }
    
    def train_models(self, X_train, y_train):
        """
        Train all models.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        """
        self.initialize_models()
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            print(f"{name} trained successfully")
        
        return self.models
    
    def cross_validate_models(self, X_train, y_train, cv=5):
        """
        Perform cross-validation on all models.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        cv : int
            Number of folds
        
        Returns:
        --------
        dict
            Cross-validation scores
        """
        cv_scores = {}
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
            cv_scores[name] = {
                'mean': scores.mean(),
                'std': scores.std()
            }
            print(f"{name} CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_scores
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all models on test set.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test target
        
        Returns:
        --------
        dict
            Evaluation results for each model
        """
        from model_evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator()
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)
            self.results[name] = metrics
            
            print(f"\n{name} Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return self.results
    
    def select_best_model(self):
        """Select best model based on ROC-AUC score."""
        best_score = 0
        
        for name, metrics in self.results.items():
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print(f"\nBest Model: {self.best_model_name} with ROC-AUC: {best_score:.4f}")
        
        return self.best_model, self.best_model_name
    
    def save_best_model(self, filepath):
        """Save the best model to file."""
        joblib.dump(self.best_model, filepath)
        print(f"Best model saved to {filepath}")
    
    def save_all_models(self, directory):
        """Save all trained models."""
        for name, model in self.models.items():
            filepath = f"{directory}/{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, filepath)
            print(f"{name} model saved to {filepath}")


def main():
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    data_path = '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/raw/credit_data.csv'
    
    preprocessor = DataPreprocessor()
    X, y, feature_cols, scaler = preprocessor.preprocess(data_path)
    
    engineer = FeatureEngineer()
    df = pd.DataFrame(X, columns=feature_cols)
    df['default'] = y
    df = engineer.create_all_features(df)
    
    X = df.drop('default', axis=1)
    y = df['default']
    
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    trainer.train_models(X_train, y_train)
    
    trainer.cross_validate_models(X_train, y_train)
    
    trainer.evaluate_models(X_test, y_test)
    
    best_model, best_name = trainer.select_best_model()
    
    model_path = '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/models/model.pkl'
    trainer.save_best_model(model_path)
    
    preprocessor.save_preprocessor('/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/processed/preprocessor.pkl')
    joblib.dump(scaler, '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/processed/scaler.pkl')


if __name__ == '__main__':
    main()
