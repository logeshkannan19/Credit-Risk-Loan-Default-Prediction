# Credit Risk & Loan Default Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

A complete end-to-end machine learning project for predicting credit risk and loan defaults. This project demonstrates data preprocessing, feature engineering, model training, evaluation, and deployment via a REST API.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [API](#api)
- [API Endpoints](#api-endpoints)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a complete machine learning pipeline for credit risk assessment:

- **Data Preprocessing**: Handle missing values, scale features using StandardScaler
- **Feature Engineering**: Create derived features like debt-to-income ratio, credit utilization
- **Model Training**: Train and compare multiple ML models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- **Model Evaluation**: Evaluate with various metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Deployment**: REST API using Flask for real-time predictions

## Features

- Modular architecture with separate components
- Configuration-based settings (YAML)
- Comprehensive logging
- Unit tests
- Makefile for common commands
- Professional package structure

## Project Structure

```
credit-risk-loan-default-prediction/
├── app/
│   └── app.py                  # Flask REST API
├── config.yaml                 # Configuration file
├── data/
│   ├── processed/              # Processed data
│   └── raw/                    # Raw dataset
├── Makefile                    # Make commands
├── models/                     # Trained models
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── README.md
├── requirements.txt
├── setup.py                    # Package setup
├── src/
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration loader
│   ├── data_generator.py       # Data generation
│   ├── data_preprocessing.py   # Data preprocessing
│   ├── feature_engineering.py # Feature engineering
│   ├── logging_config.py       # Logging setup
│   ├── model_evaluation.py     # Model evaluation
│   ├── model_training.py       # Model training
│   ├── train.py                # Main training script
│   └── utils.py                # Utility functions
└── tests/                      # Unit tests
    ├── __init__.py
    └── test_data_preprocessing.py
```

## Installation

### Option 1: Install from source

```bash
# Clone the repository
git clone https://github.com/logeshkannan19/Credit-Risk-Loan-Default-Prediction.git
cd Credit-Risk-Loan-Default-Prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Option 2: Using Make

```bash
make install
```

## Quick Start

### 1. Train the Model

```bash
python -m src.train
```

Or using Make:

```bash
make train
```

### 2. Start the API

```bash
python -m src.app
```

Or:

```bash
make api
```

The API will start on `http://localhost:5000`

### 3. Make a Prediction

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

## Usage

### Training

The training script can be run with:

```bash
python -m src.train
```

This will:
1. Load and preprocess the data
2. Engineer new features
3. Train multiple models
4. Evaluate and select the best model
5. Save the model and preprocessing artifacts

### API

Start the Flask API:

```bash
python -m src.app
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/batch_predict` | POST | Batch predictions |
| `/model_info` | GET | Model information |

### Example Response

```json
{
  "prediction": "No Default",
  "probability": 0.4235,
  "risk_level": "Medium"
}
```

### Risk Levels

- **Low**: Probability < 0.3
- **Medium**: 0.3 <= Probability < 0.6
- **High**: Probability >= 0.6

## Model Performance

The model is evaluated using multiple metrics:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.4947 | 0.6089 | 0.5014 | 0.5500 | 0.4989 |
| Decision Tree | 0.5214 | 0.6126 | 0.6061 | 0.6094 | 0.4946 |
| Random Forest | 0.5506 | 0.6166 | 0.7142 | 0.6618 | 0.5085 |
| Gradient Boosting | 0.5973 | 0.6139 | 0.9320 | 0.7403 | 0.5068 |

**Best Model**: Random Forest (selected based on ROC-AUC score)

## Configuration

All settings can be configured in `config.yaml`:

```yaml
# Training Settings
training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5

# API Settings
api:
  host: "0.0.0.0"
  port: 5000

# Risk Thresholds
risk:
  low: 0.3
  medium: 0.6
```

## Testing

Run tests with:

```bash
pytest tests/ -v
```

Or using Make:

```bash
make test
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- **Logesh Kannan** - [logeshkannan19](https://github.com/logeshkannan19)

## Acknowledgments

- Scikit-learn for machine learning utilities
- Flask for REST API
- The "Give Me Some Credit" dataset inspiration
