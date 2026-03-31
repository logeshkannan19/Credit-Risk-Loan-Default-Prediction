import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import joblib


class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, y_true, y_pred, y_proba):
        """
        Calculate all evaluation metrics.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray
            Predicted probabilities
        
        Returns:
        --------
        dict
            Dictionary of metrics
        """
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
        
        return self.metrics
    
    def get_classification_report(self, y_true, y_pred):
        """Get detailed classification report."""
        return classification_report(y_true, y_pred)
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        save_path : str
            Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Default', 'Default'],
                    yticklabels=['No Default', 'Default'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, y_true, y_proba, save_path=None):
        """
        Plot ROC curve.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_proba : np.ndarray
            Predicted probabilities
        save_path : str
            Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self, model, feature_names, save_path=None):
        """
        Plot feature importance for tree-based models.
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        feature_names : list
            List of feature names
        save_path : str
            Path to save the plot
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importances')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices],
                      rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Feature importance saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        else:
            print("Model does not have feature_importances_ attribute")
    
    def compare_models(self, results_dict, save_path=None):
        """
        Compare multiple models' performance.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary with model names as keys and metrics as values
        save_path : str
            Path to save the plot
        """
        metrics_df = pd.DataFrame(results_dict).T
        
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Model Comparison')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return metrics_df


import pandas as pd


def main():
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    from model_training import ModelTrainer
    
    data_path = '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/raw/credit_data.csv'
    model_path = '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/models/model.pkl'
    
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
    results = trainer.evaluate_models(X_test, y_test)
    
    evaluator = ModelEvaluator()
    evaluator.plot_confusion_matrix(
        y_test,
        trainer.best_model.predict(X_test),
        save_path='/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/models/confusion_matrix.png'
    )
    evaluator.plot_roc_curve(
        y_test,
        trainer.best_model.predict_proba(X_test)[:, 1],
        save_path='/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/models/roc_curve.png'
    )
    evaluator.plot_feature_importance(
        trainer.best_model,
        X.columns.tolist(),
        save_path='/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/models/feature_importance.png'
    )
    evaluator.compare_models(
        results,
        save_path='/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/models/model_comparison.png'
    )


if __name__ == '__main__':
    main()
