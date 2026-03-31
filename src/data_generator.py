import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples=10000, random_state=42):
    """
    Generate synthetic credit risk data for loan default prediction.
    
    Creates data with meaningful relationships for predicting defaults.
    """
    np.random.seed(random_state)
    
    base_features = np.random.randn(n_samples, 8)
    
    income = np.exp(base_features[:, 0] * 0.5 + 10.5)
    income = np.clip(income, 20000, 500000)
    
    credit_score = base_features[:, 1] * 80 + 650
    credit_score = np.clip(credit_score, 300, 850)
    
    debt_amount = base_features[:, 2] * 15000 + 25000
    debt_amount = np.clip(np.abs(debt_amount), 0, 100000)
    
    age = base_features[:, 3] * 12 + 40
    age = np.clip(age, 18, 80)
    
    employment_years = np.exp(base_features[:, 4]) * 2
    employment_years = np.clip(employment_years, 0, 40)
    
    loan_amount = base_features[:, 5] * 10000 + 30000
    loan_amount = np.clip(np.abs(loan_amount), 1000, 100000)
    
    interest_rate = base_features[:, 6] * 3 + 12
    interest_rate = np.clip(interest_rate, 3, 25)
    
    payment_history = base_features[:, 7] * 15 + 80
    payment_history = np.clip(payment_history, 0, 100)
    
    monthly_expenses = income * (np.random.uniform(0.2, 0.5, n_samples))
    monthly_expenses = np.clip(monthly_expenses, 500, 10000)
    
    existing_credits = np.random.randint(0, 8, n_samples)
    
    default_prob = (
        -0.3 * (credit_score - 650) / 100
        + 0.4 * (debt_amount / (income + 1))
        + 0.3 * (loan_amount / (income + 1))
        - 0.2 * (payment_history / 100)
        + 0.1 * (interest_rate - 10) / 10
        - 0.1 * (employment_years / 10)
        + 0.2 * (age - 30) / 20
        + base_features[:, 0] * 0.3
    )
    default_prob = 1 / (1 + np.exp(-default_prob))
    default = (np.random.random(n_samples) < default_prob).astype(int)
    
    df = pd.DataFrame({
        'age': age.astype(int),
        'income': income.round(2),
        'credit_score': credit_score.round(0).astype(int),
        'debt_amount': debt_amount.round(2),
        'monthly_expenses': monthly_expenses.round(2),
        'employment_years': employment_years.round(1),
        'loan_amount': loan_amount.round(2),
        'existing_credits': existing_credits,
        'interest_rate': interest_rate.round(2),
        'payment_history': payment_history.round(2),
        'default': default
    })
    
    missing_mask = np.random.random(df.shape) < 0.015
    df = df.mask(missing_mask)
    
    return df


if __name__ == '__main__':
    df = generate_synthetic_data(n_samples=15000)
    
    output_path = '/Users/lk/Documents/Projects/credit-risk-loan-default-prediction/data/raw/credit_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['default'].value_counts(normalize=True)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
