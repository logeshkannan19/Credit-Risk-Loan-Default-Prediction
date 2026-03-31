from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="credit-risk-prediction",
    version="1.0.0",
    author="Logesh Kannan",
    author_email="logeshkannan19@github.com",
    description="Credit Risk & Loan Default Prediction ML Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/logeshkannan19/Credit-Risk-Loan-Default-Prediction",
    project_urls={
        "Bug Tracker": "https://github.com/logeshkannan19/Credit-Risk-Loan-Default-Prediction/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "flask>=2.3.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "credit-risk-predict=train:main",
        ],
    },
)
