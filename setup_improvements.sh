#!/bin/bash
# Setup script for installing all dependencies

echo "=== Setting up FINQ Stock Prediction Improvements ==="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  No virtual environment detected. Please activate your virtual environment first."
    echo "Run: source venv/bin/activate"
    exit 1
fi

echo "✓ Virtual environment detected: $VIRTUAL_ENV"

# Install TA-Lib system dependency (Mac only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "\n📦 Installing TA-Lib system library (Mac)..."
    if command -v brew &> /dev/null; then
        brew install ta-lib
    else
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
fi

# Install Python packages
echo "\n📦 Installing Python packages..."
pip install --upgrade pip

# Core ML packages
echo "Installing machine learning packages..."
pip install xgboost lightgbm catboost optuna

# Technical analysis
echo "Installing technical analysis packages..."
pip install ta-lib

# Check installation
echo "\n✅ Checking installations..."
python -c "import xgboost; print('✓ XGBoost installed')"
python -c "import lightgbm; print('✓ LightGBM installed')"
python -c "import catboost; print('✓ CatBoost installed')"
python -c "import optuna; print('✓ Optuna installed')"
python -c "import talib; print('✓ TA-Lib installed')"

echo "\n🎉 Setup complete! You can now run:"
echo "   python train_models.py --use_enhanced_features"
echo "   python test_improvements.py" 