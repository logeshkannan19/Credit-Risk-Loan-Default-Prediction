import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
    
    def load_data(self, filepath):
        """Load data from CSV file."""
        df = pd.read_csv(filepath)
        return df
    
    def handle_missing_values(self, df, target_col='default'):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with missing values handled
        """
        df = df.dropna(subset=[target_col])
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        return df
    
    def remove_outliers(self, df, columns, z_threshold=3):
        """
        Remove outliers using Z-score method.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            Columns to check for outliers
        z_threshold : float
            Z-score threshold for outlier removal
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean = df_clean[z_scores < z_threshold]
        
        return df_clean
    
    def split_features_target(self, df, target_col='default'):
        """
        Split dataframe into features and target.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column
        
        Returns:
        --------
        tuple
            (X, y) features and target
        """
        self.feature_columns = [col for col in df.columns if col != target_col]
        
        X = df[self.feature_columns]
        y = df[target_col]
        
        return X, y
    
    def scale_features(self, X, fit=True):
        """
        Scale features using StandardScaler.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe
        fit : bool
            Whether to fit the scaler
        
        Returns:
        --------
        np.ndarray
            Scaled features
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def preprocess(self, filepath, remove_outliers=False):
        """
        Complete preprocessing pipeline.
        
        Parameters:
        -----------
        filepath : str
            Path to input data
        remove_outliers : bool
            Whether to remove outliers
        
        Returns:
        --------
        tuple
            (X_scaled, y, feature_columns, scaler)
        """
        df = self.load_data(filepath)
        
        df = self.handle_missing_values(df)
        
        if remove_outliers:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols.remove('default')
            df = self.remove_outliers(df, numeric_cols)
        
        X, y = self.split_features_target(df)
        
        X_scaled = self.scale_features(X, fit=True)
        
        return X_scaled, y, self.feature_columns, self.scaler
    
    def save_preprocessor(self, filepath):
        """Save preprocessor to file."""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load_preprocessor(filepath):
        """Load preprocessor from file."""
        return joblib.load(filepath)


def main():
    preprocessor = DataPreprocessor()
    
    input_path = '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/raw/credit_data.csv'
    output_path = '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/processed/'
    
    X, y, feature_cols, scaler = preprocessor.preprocess(input_path, remove_outliers=False)
    
    print(f"Preprocessed data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {feature_cols}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    preprocessor.save_preprocessor(output_path + 'preprocessor.pkl')
    print(f"Preprocessor saved to {output_path}preprocessor.pkl")


if __name__ == '__main__':
    main()
