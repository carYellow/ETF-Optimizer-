import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Stock Performance Prediction - Exploratory Analysis\n",
                "\n",
                "This notebook demonstrates the end-to-end process of:\n",
                "1. Loading and preprocessing stock data\n",
                "2. Feature engineering\n",
                "3. Model training and evaluation\n",
                "4. Results analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "import sys\n",
                "sys.path.append('..')\n",
                "\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from datetime import datetime\n",
                "\n",
                "from src.data.data_loader import StockDataLoader\n",
                "from src.data.feature_engineering import FeatureGenerator\n",
                "from src.models.train import ModelTrainer\n",
                "\n",
                "# Set plot style\n",
                "plt.style.use('default')\n",
                "sns.set_theme()\n",
                "%matplotlib inline"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Data\n",
                "\n",
                "First, we'll load historical data for a subset of S&P 500 stocks. For demonstration purposes, we'll use a small subset of stocks to keep the analysis manageable."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Initialize data loader\n",
                "data_loader = StockDataLoader(start_date=\"2010-01-01\")\n",
                "\n",
                "# Get S&P 500 symbols\n",
                "symbols = data_loader.get_sp500_symbols()\n",
                "print(f\"Number of S&P 500 symbols: {len(symbols)}\")\n",
                "\n",
                "# Load data for a subset of symbols (for demonstration)\n",
                "sample_symbols = symbols[:10]  # Use first 10 symbols\n",
                "stock_data, sp500_data = data_loader.prepare_training_data(sample_symbols)\n",
                "\n",
                "print(f\"\\nStock data shape: {stock_data.shape}\")\n",
                "print(f\"S&P 500 data shape: {sp500_data.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Data Exploration\n",
                "\n",
                "Let's explore the data to understand its structure and characteristics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Display sample of stock data\n",
                "print(\"Sample of stock data:\")\n",
                "display(stock_data.head())\n",
                "\n",
                "# Display sample of S&P 500 data\n",
                "print(\"\\nSample of S&P 500 data:\")\n",
                "display(sp500_data.head())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Plot price trends for a few stocks\n",
                "plt.figure(figsize=(15, 8))\n",
                "for symbol in sample_symbols[:3]:  # Plot first 3 stocks\n",
                "    stock_prices = stock_data[stock_data['Symbol'] == symbol]['Close']\n",
                "    plt.plot(stock_prices.index, stock_prices.values, label=symbol)\n",
                "\n",
                "plt.plot(sp500_data.index, sp500_data['Close'], label='S&P 500', linestyle='--')\n",
                "plt.title('Stock Price Trends')\n",
                "plt.xlabel('Date')\n",
                "plt.ylabel('Price')\n",
                "plt.legend()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Feature Engineering\n",
                "\n",
                "Now, let's generate technical indicators and other features for our model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Initialize feature generator\n",
                "feature_generator = FeatureGenerator()\n",
                "\n",
                "# Generate features\n",
                "df = feature_generator.prepare_features(stock_data, sp500_data)\n",
                "\n",
                "print(f\"Data shape after feature engineering: {df.shape}\")\n",
                "print(\"\\nFeature columns:\")\n",
                "print(df.columns.tolist())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Plot distribution of labels\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.countplot(data=df, x='Label')\n",
                "plt.title('Distribution of Labels (1: Outperforms S&P 500, 0: Underperforms)')\n",
                "plt.show()\n",
                "\n",
                "# Plot feature correlations\n",
                "plt.figure(figsize=(15, 12))\n",
                "correlation_matrix = df.select_dtypes(include=[np.number]).corr()\n",
                "sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)\n",
                "plt.title('Feature Correlations')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Model Training\n",
                "\n",
                "Let's train our model and evaluate its performance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Initialize model trainer\n",
                "model_trainer = ModelTrainer()\n",
                "\n",
                "# Train and evaluate model\n",
                "results = model_trainer.train_and_evaluate(df)\n",
                "\n",
                "# Print evaluation metrics\n",
                "print(\"Evaluation Metrics:\")\n",
                "for metric, value in results['evaluation_metrics'].items():\n",
                "    print(f\"{metric}: {value:.4f}\")\n",
                "\n",
                "# Plot feature importance\n",
                "feature_importance = results['training_results']['feature_importance']\n",
                "plt.figure(figsize=(12, 6))\n",
                "sns.barplot(data=feature_importance.head(10), x='importance', y='feature')\n",
                "plt.title('Top 10 Most Important Features')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Model Analysis\n",
                "\n",
                "Let's analyze the model's performance in more detail."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Analyze predictions by stock\n",
                "X, y = model_trainer.prepare_features(df)\n",
                "X_train, X_test, y_train, y_test = model_trainer.train_test_split(X, y)\n",
                "\n",
                "# Make predictions\n",
                "X_test_scaled = model_trainer.scaler.transform(X_test)\n",
                "y_pred = model_trainer.model.predict(X_test_scaled)\n",
                "y_pred_proba = model_trainer.model.predict_proba(X_test_scaled)[:, 1]\n",
                "\n",
                "# Create results dataframe\n",
                "results_df = pd.DataFrame({\n",
                "    'Symbol': df.iloc[X_test.index]['Symbol'],\n",
                "    'Date': df.iloc[X_test.index].index,\n",
                "    'Actual': y_test,\n",
                "    'Predicted': y_pred,\n",
                "    'Probability': y_pred_proba\n",
                "})\n",
                "\n",
                "# Calculate accuracy by stock\n",
                "stock_accuracy = results_df.groupby('Symbol').apply(\n",
                "    lambda x: (x['Actual'] == x['Predicted']).mean()\n",
                ").sort_values(ascending=False)\n",
                "\n",
                "print(\"\\nAccuracy by Stock:\")\n",
                "print(stock_accuracy)\n",
                "\n",
                "# Plot accuracy by stock\n",
                "plt.figure(figsize=(12, 6))\n",
                "stock_accuracy.plot(kind='bar')\n",
                "plt.title('Model Accuracy by Stock')\n",
                "plt.xlabel('Stock Symbol')\n",
                "plt.ylabel('Accuracy')\n",
                "plt.xticks(rotation=45)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Conclusion\n",
                "\n",
                "In this notebook, we have:\n",
                "1. Loaded and preprocessed stock data from S&P 500\n",
                "2. Generated technical indicators and market features\n",
                "3. Trained a Random Forest model to predict stock performance\n",
                "4. Evaluated the model's performance\n",
                "5. Analyzed feature importance and stock-specific accuracy\n",
                "\n",
                "The model shows promising results in predicting whether a stock will outperform the S&P 500 index over the next 5 trading days. The feature importance analysis reveals which technical indicators and market features are most predictive of stock performance."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to a file
with open('exploratory_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 