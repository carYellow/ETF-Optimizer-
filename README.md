# Stock Performance Prediction System

This project implements an end-to-end machine learning pipeline for predicting whether a given S&P 500 stock will outperform the S&P 500 index over the next 5 trading days.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   └── raw/                    # Raw data from Yahoo Finance
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Data fetching and preprocessing
│   │   └── feature_engineering.py  # Feature generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py           # Model training pipeline
│   │   └── predict.py         # Model inference
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py             # FastAPI application
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Utility functions
├── notebooks/
│   └── exploratory_analysis.ipynb  # Jupyter notebook for analysis
└── tests/
    └── __init__.py
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
uvicorn src.api.app:app --reload
```

## Features

- Historical OHLCV data processing from Yahoo Finance
- Feature engineering for stock price prediction
- Machine learning model training with proper train/test split
- REST API for model inference
- Modular and maintainable codebase

## API Usage

The API exposes the following endpoints:

- `POST /predict`: Predict stock performance
  - Input: Stock symbol and date
  - Output: Prediction and confidence score

Example request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "AAPL", "date": "2024-03-20"}'
```

## MLOps Plan

The project includes a plan for MLOps implementation in `mlops_plan.md`, covering:
- Docker containerization
- MLflow for experiment tracking
- CI/CD pipeline setup
- Monitoring and logging strategy

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest tests/
```

## License

MIT License 