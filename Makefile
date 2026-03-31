.PHONY: help install train test clean run api lint format

help:
	@echo "Credit Risk & Loan Default Prediction - Make Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train the model"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean generated files"
	@echo "  make run        - Run the training script directly"
	@echo "  make api        - Start Flask API"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code with black"

install:
	pip install -r requirements.txt
	pip install -e .

train:
	python -m src.train

run:
	python -m src.train

api:
	python -m src.app

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -f models/*.pkl models/*.png 2>/dev/null || true

lint:
	flake8 src/ --max-line-length=100 --ignore=E203,W503

format:
	black src/ --line-length=100

eda:
	jupyter notebook notebooks/eda.ipynb
