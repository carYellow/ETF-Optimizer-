# FINQ Stock Prediction Project

A machine learning pipeline for predicting whether a given S&P 500 stock will outperform the index over the next 5 trading days.

## 🚀 Recent Improvements

### 1. Enhanced Feature Engineering (100+ features)
- Advanced technical indicators (Bollinger Bands, MACD, ADX, etc.)
- Volume-based indicators (OBV, Money Flow Index, etc.)
- Volatility measures (Parkinson, Garman-Klass)
- Candlestick pattern recognition
- Market microstructure features
- Market regime indicators

### 2. Robust Train/Test Splitting
- **Lookahead bias prevention** with temporal gaps
- **Event-aware splitting** avoiding major market events
- **Walk-forward analysis** for robust backtesting
- **Expanding window** validation

### 3. Advanced Model Selection
- Multiple algorithms: XGBoost, LightGBM, CatBoost, Random Forest, SVM
- Hyperparameter optimization with Optuna
- Ensemble methods
- Feature selection techniques

## 📋 Requirements

- Python 3.8+
- Dependencies in `requirements.txt`
- TA-Lib (see installation guide below)

## 🛠️ Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd finq-stock-prediction
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install TA-Lib (required for enhanced features):
```bash
# Mac
brew install ta-lib

# Linux
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## 🏃 Quick Start

### Basic Training
```bash
# Train with default settings
python train_models.py
```

### Enhanced Training
```bash
# Use all enhancements
python train_models.py --use_enhanced_features --split_method walk_forward --optimization optuna
```

### Feature Analysis
```bash
# Run comprehensive feature analysis
cd notebooks
python feature_engineering_eda.py
```

## 📊 Project Structure

```
├── src/
│   ├── data/
│   │   ├── data_loader.py          # Data fetching from Yahoo Finance
│   │   ├── feature_engineering.py  # Basic feature generation
│   │   ├── enhanced_features.py    # Advanced features (NEW)
│   │   └── train_test_split.py     # Robust splitting methods (NEW)
│   ├── models/
│   │   ├── train.py                # Basic model training
│   │   ├── advanced_train.py       # Multi-model training (NEW)
│   │   └── predict.py              # Prediction interface
│   └── api/
│       └── app.py                  # FastAPI application
├── notebooks/
│   ├── exploratory_analysis.ipynb  # Initial EDA
│   └── feature_engineering_eda.py  # Feature analysis (NEW)
├── docs/
│   └── training_guide.md           # Comprehensive guide (NEW)
├── train_models.py                 # Main training script (NEW)
└── requirements.txt                # Updated dependencies
```

## 🎯 API Usage

Start the API server:
```bash
uvicorn src.api.app:app --reload
```

Make predictions:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"stock_id": "AAPL", "date": "2024-01-15"}'
```

## 📈 Model Performance

Expected performance metrics:
- **ROC AUC**: 0.55-0.65 (better than random)
- **Accuracy**: 55-60%
- **Best Models**: XGBoost, LightGBM typically perform best

## 🔧 Advanced Configuration

See `docs/training_guide.md` for detailed configuration options including:
- Feature selection methods
- Hyperparameter optimization strategies
- Cross-validation approaches
- Ensemble techniques

## 📝 MLOps Plan

See `mlops_plan.md` for production deployment strategy including:
- Docker containerization
- Model versioning with MLflow
- Automated retraining pipeline
- Performance monitoring

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📄 License

This project is proprietary to FINQ. 