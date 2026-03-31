import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor."""
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample data for testing."""
        data = {
            'age': [25, 35, 45, 55, np.nan],
            'income': [50000, 60000, 70000, np.nan, 90000],
            'credit_score': [650, 700, 750, 800, 850],
            'debt_amount': [5000, 10000, 15000, 20000, 25000],
            'monthly_expenses': [1500, 2000, 2500, 3000, 3500],
            'employment_years': [2, 5, 10, 15, 20],
            'loan_amount': [10000, 20000, 30000, 40000, 50000],
            'existing_credits': [1, 2, 3, 4, 5],
            'interest_rate': [10.0, 12.0, 14.0, 16.0, 18.0],
            'payment_history': [80, 85, 90, 95, 100],
            'default': [0, 0, 1, 1, 0]
        }
        df = pd.DataFrame(data)
        filepath = tmp_path / "test_data.csv"
        df.to_csv(filepath, index=False)
        return filepath
    
    def test_load_data(self, sample_data):
        """Test loading data from CSV."""
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(str(sample_data))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
    
    def test_handle_missing_values(self, sample_data):
        """Test handling missing values."""
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(str(sample_data))
        df_clean = preprocessor.handle_missing_values(df)
        
        assert df_clean['default'].isna().sum() == 0
        assert df_clean['age'].isna().sum() == 0
        assert df_clean['income'].isna().sum() == 0
    
    def test_preprocess_pipeline(self, sample_data):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor()
        X, y, feature_cols, scaler = preprocessor.preprocess(str(sample_data))
        
        assert len(feature_cols) == 10
        assert len(X) == len(y)
        assert X.shape[1] == 10
