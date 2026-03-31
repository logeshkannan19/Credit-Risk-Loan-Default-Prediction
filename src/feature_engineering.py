import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self):
        self.new_features = None
    
    def create_debt_to_income_ratio(self, df):
        """
        Create Debt-to-Income ratio feature.
        
        DTI = Monthly Debt Payments / Monthly Income
        """
        df = df.copy()
        df['debt_to_income_ratio'] = df['debt_amount'] / (df['income'] / 12 + 1)
        return df
    
    def create_credit_utilization(self, df):
        """
        Create Credit Utilization ratio feature.
        """
        df = df.copy()
        df['credit_utilization'] = df['debt_amount'] / (df['credit_score'] + 1)
        return df
    
    def create_affordability_score(self, df):
        """
        Create affordability score based on income vs expenses.
        """
        df = df.copy()
        df['affordability_score'] = (df['income'] - df['monthly_expenses'] * 12) / (df['income'] + 1)
        return df
    
    def create_loan_to_income_ratio(self, df):
        """
        Create Loan-to-Income ratio.
        """
        df = df.copy()
        df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)
        return df
    
    def create_payment_to_income_ratio(self, df):
        """
        Create monthly payment to income ratio.
        Assumes simple monthly payment calculation.
        """
        df = df.copy()
        df['monthly_payment'] = df['loan_amount'] * (df['interest_rate'] / 100 / 12 + 1)
        df['payment_to_income_ratio'] = df['monthly_payment'] / (df['income'] / 12 + 1)
        df = df.drop('monthly_payment', axis=1)
        return df
    
    def create_employment_stability(self, df):
        """
        Create employment stability score.
        """
        df = df.copy()
        df['employment_stability'] = df['employment_years'] / (df['age'] + 1)
        return df
    
    def create_credit_history_score(self, df):
        """
        Create credit history score based on payment history and existing credits.
        """
        df = df.copy()
        df['credit_history_score'] = df['payment_history'] * (1 / (df['existing_credits'] + 1))
        return df
    
    def create_all_features(self, df):
        """
        Apply all feature engineering transformations.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with new features
        """
        df = self.create_debt_to_income_ratio(df)
        df = self.create_credit_utilization(df)
        df = self.create_affordability_score(df)
        df = self.create_loan_to_income_ratio(df)
        df = self.create_payment_to_income_ratio(df)
        df = self.create_employment_stability(df)
        df = self.create_credit_history_score(df)
        
        self.new_features = [
            'debt_to_income_ratio',
            'credit_utilization',
            'affordability_score',
            'loan_to_income_ratio',
            'payment_to_income_ratio',
            'employment_stability',
            'credit_history_score'
        ]
        
        return df
    
    def select_features(self, df, target_col='default', top_n=15):
        """
        Select top features based on correlation with target.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Target column name
        top_n : int
            Number of top features to select
        
        Returns:
        --------
        list
            List of selected feature names
        """
        correlations = df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
        selected_features = correlations.head(top_n).index.tolist()
        
        return selected_features


def main():
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    input_path = '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/raw/credit_data.csv'
    
    df = preprocessor.load_data(input_path)
    df = preprocessor.handle_missing_values(df)
    
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    
    print(f"Original features: 10")
    print(f"New features created: {len(engineer.new_features)}")
    print(f"Total features: {len(df.columns) - 1}")
    print(f"\nNew features: {engineer.new_features}")
    
    selected = engineer.select_features(df, top_n=15)
    print(f"\nTop 15 features by correlation: {selected}")
    
    output_path = '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/processed/credit_data_engineered.csv'
    df.to_csv(output_path, index=False)
    print(f"\nEngineered data saved to {output_path}")


if __name__ == '__main__':
    main()
