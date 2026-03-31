# Credit Risk & Loan Default Prediction

A complete end-to-end machine learning project for predicting credit risk and loan defaults. This project demonstrates data preprocessing, feature engineering, model training, evaluation, and deployment via a Flask REST API.

## Project Overview

This project implements a complete machine learning pipeline for credit risk assessment:

- **Data Preprocessing**: Handle missing values, scale features
- **Feature Engineering**: Create derived features like debt-to-income ratio, credit utilization
- **Model Training**: Train and compare multiple ML models
- **Model Evaluation**: Evaluate with various metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Deployment**: REST API for real-time predictions

## Tech Stack

- **Language**: Python 3.9+
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **API**: Flask
- **Serialization**: joblib

## Project Structure

```
credit-risk-loan-default-prediction/
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Processed data
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── src/
│   ├── data_preprocessing.py   # Data preprocessing module
│   ├── feature_engineering.py # Feature engineering module
│   ├── model_training.py       # Model training module
│   ├── model_evaluation.py     # Model evaluation module
│   └── train.py                # Main training script
├── models/
│   └── model.pkl               # Trained model
├── app/
│   └── app.py                  # Flask API
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Setup Instructions

1. **Clone the repository** (if applicable) or navigate to the project directory:

```bash
cd credit-risk-loan-default-prediction
```

2. **Create a virtual environment** (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Generate the dataset** (if not already present):

```bash
python3 src/data_generator.py
```

5. **Train the model**:

```bash
python3 src/train.py
```

## How to Run the API

After training the model, start the Flask API:

```bash
python3 app/app.py
```

The API will start on `http://0.0.0.0:5000`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/batch_predict` | POST | Batch predictions |
| `/model_info` | GET | Model information |

## Example Request/Response

### Single Prediction

**Request:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**

```json
{
  "prediction": "No Default",
  "probability": 0.4235,
  "risk_level": "Medium"
}
```

### Batch Prediction

**Request:**

```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "applicants": [
      {"age": 35, "income": 75000, "credit_score": 650, "debt_amount": 15000, "monthly_expenses": 2500, "employment_years": 5, "loan_amount": 25000, "existing_credits": 2, "interest_rate": 12.5, "payment_history": 85},
      {"age": 45, "income": 60000, "credit_score": 580, "debt_amount": 25000, "monthly_expenses": 2000, "employment_years": 10, "loan_amount": 30000, "existing_credits": 3, "interest_rate": 15.0, "payment_history": 70}
    ]
  }'
```

**Response:**

```json
{
  "predictions": [
    {"prediction": "No Default", "probability": 0.4235, "risk_level": "Medium"},
    {"prediction": "Default", "probability": 0.6821, "risk_level": "High"}
  ]
}
```

## Model Performance

The model is evaluated using multiple metrics:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.4947 | 0.6089 | 0.5014 | 0.5500 | 0.4989 |
| Decision Tree | 0.5214 | 0.6126 | 0.6061 | 0.6094 | 0.4946 |
| Random Forest | 0.5506 | 0.6166 | 0.7142 | 0.6618 | 0.5085 |
| Gradient Boosting | 0.5973 | 0.6139 | 0.9320 | 0.7403 | 0.5068 |

**Best Model**: Random Forest (selected based on ROC-AUC score)

### Risk Levels

- **Low**: Probability < 0.3
- **Medium**: 0.3 <= Probability < 0.6
- **High**: Probability >= 0.6

## Features

### Original Features

1. `age` - Applicant age
2. `income` - Annual income
3. `credit_score` - Credit score (300-850)
4. `debt_amount` - Existing debt
5. `monthly_expenses` - Monthly expenses
6. `employment_years` - Years of employment
7. `loan_amount` - Requested loan amount
8. `existing_credits` - Number of existing credits
9. `interest_rate` - Loan interest rate
10. `payment_history` - Payment history score (0-100)

### Engineered Features

1. `debt_to_income_ratio` - Debt to income ratio
2. `credit_utilization` - Credit utilization ratio
3. `affordability_score` - Affordability score
4. `loan_to_income_ratio` - Loan to income ratio
5. `payment_to_income_ratio` - Payment to income ratio
6. `employment_stability` - Employment stability score
7. `credit_history_score` - Credit history score

## License

MIT License
