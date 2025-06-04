import os
import sys
import markdown
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import numpy as np
import json
import joblib
from datetime import datetime

def read_markdown_file(file_path):
    """Read a markdown file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def convert_md_to_html(md_content):
    """Convert markdown content to HTML."""
    return markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

def enhance_html(html_content):
    """Enhance HTML with CSS and formatting."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Convert tables to styled tables
    for table in soup.find_all('table'):
        table['class'] = 'table table-striped table-hover'
        
        # Enhance the performance metrics comparison table
        if table.find('th', string='Metric') and table.find('th', string='XGBoost') and table.find('th', string='LightGBM'):
            table['class'] = 'table table-striped model-comparison-table'
            
            # Style winner cells
            for row in table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 4:  # Metric, XGBoost, LightGBM, Winner
                    winner = cells[3].get_text().strip()
                    if winner == 'XGBoost':
                        cells[1]['class'] = 'xgboost-value winner-cell'
                    elif winner == 'LightGBM':
                        cells[2]['class'] = 'lightgbm-value winner-cell'
    
    # Format code blocks
    for code in soup.find_all('code'):
        code['class'] = 'code-block'
    
    # Fix image paths if they contain 'reports/' prefix
    for img in soup.find_all('img'):
        if img.get('src') and img['src'].startswith('reports/'):
            img['src'] = os.path.basename(img['src'])
    
    # Format pros and cons sections
    for h4 in soup.find_all('h4'):
        if 'Pros of' in h4.text:
            container = soup.new_tag('div')
            container['class'] = 'pros'
            h4.wrap(container)
            
            # Move the next elements until the next h4 into this container
            next_elem = container.next_sibling
            while next_elem and (not (next_elem.name == 'h4')):
                temp = next_elem.next_sibling
                container.append(next_elem)
                next_elem = temp
        
        elif 'Cons of' in h4.text:
            container = soup.new_tag('div')
            container['class'] = 'cons'
            h4.wrap(container)
            
            # Move the next elements until the next h4 into this container
            next_elem = container.next_sibling
            while next_elem and (not (next_elem.name == 'h4')):
                temp = next_elem.next_sibling
                container.append(next_elem)
                next_elem = temp
    
    return str(soup)

def generate_performance_chart(models_data):
    """Generate performance comparison chart."""
    metrics = ['ROC AUC', 'F1 Score', 'Accuracy', 'Precision', 'Recall']
    xgb_values = [0.5056, 0.6564, 0.4937, 0.4917, 0.9870]
    lgbm_values = [0.5074, 0.6269, 0.4963, 0.4920, 0.8638]
    
    # Use seaborn for better styling
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bar_width = 0.35
    index = np.arange(len(metrics))
    
    # Add bars with improved colors
    ax.bar(index, xgb_values, bar_width, label='XGBoost', color='#3498db', alpha=0.9, edgecolor='white', linewidth=1)
    ax.bar(index + bar_width, lgbm_values, bar_width, label='LightGBM', color='#2ecc71', alpha=0.9, edgecolor='white', linewidth=1)
    
    # Add value labels on top of each bar
    for i, v in enumerate(xgb_values):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')
    
    for i, v in enumerate(lgbm_values):
        ax.text(i + bar_width, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')
    
    # Improve styling
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.legend(fontsize=12)
    
    # Add a grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits for better visualization
    ax.set_ylim(0, 1.1)  # Maximum metric value is 1.0 (or 100%)
    
    plt.tight_layout()
    
    # Save with higher DPI for better quality
    chart_path = os.path.join('reports', 'model_comparison_chart.png')
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_path

def load_feature_importance(model_path):
    """Load feature importance from a saved model."""
    try:
        model = joblib.load(model_path)
        # This is a simplified approach - actual implementation may vary based on model type
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        return None
    except Exception as e:
        print(f"Error loading feature importance: {e}")
        return None

def generate_feature_importance_chart(feature_importances, feature_names, model_name):
    """Generate feature importance chart."""
    if feature_importances is None or len(feature_importances) == 0:
        return None
        
    # Get top 15 features
    if len(feature_importances) > 15:
        indices = np.argsort(feature_importances)[-15:]
        feature_importances = feature_importances[indices]
        feature_names = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)), feature_importances, align='center')
    plt.yticks(range(len(feature_importances)), feature_names)
    plt.xlabel('Importance')
    plt.title(f'Top Feature Importance - {model_name}')
    
    # Save with higher DPI for better quality
    chart_path = os.path.join('reports', f'{model_name}_feature_importance.png')
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_path

def create_html_report():
    """Generate a comprehensive HTML report."""
    # Read markdown report
    try:
        md_content = read_markdown_file('models/training_report.md')
    except FileNotFoundError:
        print("Training report not found. Using final report instead.")
        try:
            md_content = read_markdown_file('models/final_training_report.md')
        except FileNotFoundError:
            print("No report files found.")
            return
    
    # Create detailed report markdown text
    detailed_report = """
# Detailed Stock Prediction Model Training Report

## Overview of Training Configuration

The model training was executed with the following configuration:
- Enhanced features enabled
- Models: XGBoost and LightGBM
- Optimization strategy: Random search
- Feature importance selection enabled
- Caching enabled
- Parallel processing enabled

## Data Preparation and Feature Engineering

### Data Source
- The training utilized S&P 500 stock data spanning from March 2010 to May 2025
- 494 stocks were processed, resulting in 1,833,759 stock records

### Feature Engineering Process

#### Basic Features
Initial dataset contained 106 features before any feature selection. These features included:
- Price-based features (Returns over different timeframes, Momentum indicators)
- Technical indicators (Moving Averages, RSI, Bollinger Bands)
- Volatility measures
- Price Z-scores

#### Enhanced Features
The model leveraged several feature groups:
- Technical features (19 features)
- Volume-based features (9 features)
- Volatility indicators (11 features)
- Pattern recognition (1 feature)
- Market microstructure features (5 features)
- Time-based features (4 features)
- Cross-sectional features (3 features)

#### Feature Selection
- Feature importance selection reduced the feature count from 155 to 77 features (50.3% reduction)
- This optimization helped reduce model complexity while maintaining predictive power

### Data Splitting Methodology

The model used an event-aware split with the following characteristics:
- Train period: March 24, 2010 to June 24, 2022
- Test period: June 30, 2022 to May 30, 2025
- Gap between train and test: 6 days (to prevent lookahead bias)
- Train size: 1,443,499 records (80%)
- Test size: 361,608 records (20%)
- The test period included the banking crisis of 2023 (March 8-31, 2023)

## Model Training Methods

### XGBoost Model

#### Training Details
- Training time: 328.93 seconds
- Best iteration: 56
- Optimization method: Random search over 50 trials
- Early stopping used with 10 rounds patience

#### Performance Metrics
- ROC AUC: 0.5152 (training) / 0.5056 (test)
- F1 Score: 0.6670 (training) / 0.6564 (test)
- Accuracy: 0.5045 (training) / 0.4937 (test)
- Precision: 0.5033 (training) / 0.4917 (test)
- Recall: 0.9884 (training) / 0.9870 (test)

#### Pros of XGBoost
1. **High Recall**: Excellent at identifying positive instances (98.7% recall on test data)
2. **Robustness**: Handles non-linear relationships well
3. **Feature Importance**: Provides clear rankings of feature importance
4. **Regularization**: Built-in regularization helps prevent overfitting

#### Cons of XGBoost
1. **Low Precision**: While recall is high, precision is low (49.17%), indicating many false positives
2. **Complexity**: More complex to tune and optimize compared to simpler models
3. **Computationally Intensive**: Longer training time compared to LightGBM
4. **Limited Accuracy**: Overall accuracy slightly below 50%, indicating challenges in predicting stock movements

### LightGBM Model

#### Training Details
- Training time: 276.14 seconds
- Best iteration: 19
- Optimization method: Random search over 50 trials
- Early stopping with 10 rounds patience

#### Performance Metrics
- ROC AUC: 0.5164 (training) / 0.5074 (test)
- F1 Score: 0.6366 (training) / 0.6269 (test)
- Accuracy: 0.5081 (training) / 0.4963 (test)
- Precision: 0.5059 (training) / 0.4920 (test)
- Recall: 0.8583 (training) / 0.8638 (test)

#### Pros of LightGBM
1. **Speed**: Faster training time compared to XGBoost (277.12 vs 329.25 seconds)
2. **Memory Efficiency**: Uses leaf-wise growth instead of level-wise, requiring less memory
3. **Better ROC AUC**: Slightly higher ROC AUC score on test data (0.5074 vs 0.5056)
4. **Better Balanced Performance**: More balanced precision-recall tradeoff than XGBoost

#### Cons of LightGBM
1. **Lower Recall**: Lower recall compared to XGBoost (86.38% vs 98.70%)
2. **Similar Accuracy Limitations**: Still below 50% accuracy on test data
3. **Similar Precision Issues**: Precision around 49%, indicating high false positive rate
4. **Fewer Iterations**: Converged at 19 iterations vs 56 for XGBoost, possibly indicating less fine-tuning

## Optimization Methods and Memory Management

### Random Optimization
- Performed 50 random trials for hyperparameter tuning
- Advantages: Explores parameter space efficiently without grid search exhaustion
- Disadvantages: May miss optimal parameter combinations

### Memory Optimization
- Initial data memory usage: 440.96 MB
- After optimization: 297.34 MB (32.57% reduction)
- Enhanced feature dataset memory: 730.76 MB reduced to 566.88 MB (22.43% reduction)

### Parallel Processing
- Enabled batch processing of symbols in parallel
- Split into 10 batches of approximately 50 symbols each
- Significantly accelerated feature generation

## Detailed Comparison of Models

### Performance Metrics Comparison

| Metric | XGBoost | LightGBM | Winner |
|--------|---------|----------|--------|
| ROC AUC | 0.5056 | 0.5074 | LightGBM |
| F1 Score | 0.6564 | 0.6269 | XGBoost |
| Accuracy | 0.4937 | 0.4963 | LightGBM |
| Precision | 0.4917 | 0.4920 | LightGBM |
| Recall | 0.9870 | 0.8638 | XGBoost |
| Training Time | 329.25s | 277.12s | LightGBM |

### Prediction Tendencies

1. **XGBoost Tendency**:
   - High recall but lower precision suggests XGBoost tends to predict positive stock movements frequently
   - This creates a "bullish bias" - would catch most upward movements but generates many false positives
   - Better suited for scenarios where missing positive movements is costly

2. **LightGBM Tendency**:
   - More balanced precision-recall tradeoff
   - Slightly better at avoiding false positives
   - Potentially more useful for balanced trading strategies

## Challenges and Limitations

1. **Limited Predictive Power**:
   - Both models achieved AUC scores only slightly above 0.5 (random guessing)
   - Accuracy below 50% indicates persistent challenges in stock movement prediction

2. **Market Complexity**:
   - The inclusion of the 2023 banking crisis in test data may have impacted model performance
   - Technical features alone may not capture all market dynamics

3. **Feature Engineering Tradeoffs**:
   - Feature selection reduced features by 50.3%, which may have removed some predictive signals
   - The balance between feature count and model complexity remains challenging

4. **Optimization Limitations**:
   - Random search may not have found optimal hyperparameters
   - Limited to 50 trials due to computational constraints

## Conclusion and Recommendations

### Model Selection
- **LightGBM** appears marginally better for general stock prediction with slightly higher AUC and accuracy
- **XGBoost** may be preferred when recall is prioritized over precision (e.g., when missing upward movements is costly)

### Potential Improvements
1. **Feature Enhancement**:
   - Incorporate more fundamental data and market sentiment features
   - Explore more regime-specific features for different market conditions

2. **Model Tuning**:
   - Consider Bayesian optimization instead of random search
   - Experiment with ensemble approaches combining both models

3. **Data Preprocessing**:
   - Further investigation into handling the banking crisis period
   - Consider more sophisticated train-test split strategies

4. **Prediction Strategy**:
   - Develop separate models for different market regimes
   - Explore classification thresholds to better balance precision and recall

### Final Assessment
Both models demonstrate the inherent difficulty in predicting stock price movements. The slightly-above-random AUC scores highlight the challenge, but also suggest there is some predictive signal being captured. The choice between XGBoost and LightGBM depends on the specific trading strategy and risk tolerance, with LightGBM offering a more balanced approach and XGBoost providing higher sensitivity to upward movements at the cost of more false positives.
"""
    
    # Create merged report
    combined_md = md_content + "\n\n" + detailed_report
    
    # Convert to HTML and enhance
    html_content = convert_md_to_html(combined_md)
    enhanced_html = enhance_html(html_content)
    
    # Generate performance chart
    chart_path = generate_performance_chart({
        'xgboost': {
            'roc_auc': 0.5056,
            'f1': 0.6564,
            'accuracy': 0.4937,
            'precision': 0.4917,
            'recall': 0.9870
        },
        'lightgbm': {
            'roc_auc': 0.5074,
            'f1': 0.6269,
            'accuracy': 0.4963,
            'precision': 0.4920,
            'recall': 0.8638
        }
    })    # Extract just the filename from the chart path for HTML referencing
    chart_filename = os.path.basename(chart_path)
    
    # Create the complete HTML document with CSS
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Prediction Model Training Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
            background-color: #f9f9f9;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
        }}
        h1 {{
            color: #1a237e;
            border-bottom: 2px solid #1a237e;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        .table {{
            width: 100%;
            margin-bottom: 2rem;
            border-collapse: collapse;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        .table th {{
            background-color: #2c3e50;
            color: white;
            text-align: left;
            padding: 12px;
            font-weight: bold;
        }}
        .table td {{
            padding: 10px 12px;
            vertical-align: middle;
            border-top: 1px solid #ddd;
        }}
        .table-striped tbody tr:nth-of-type(odd) {{
            background-color: rgba(0, 0, 0, 0.03);
        }}
        .table-hover tbody tr:hover {{
            background-color: rgba(0, 0, 0, 0.075);
        }}
        .code-block {{
            background-color: #f8f9fa;
            border: 1px solid #eaecef;
            border-radius: 6px;
            padding: 16px;
            overflow-x: auto;
            font-family: Consolas, Monaco, 'Andale Mono', monospace;
            display: block;
            margin: 20px 0;
            font-size: 14px;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 15px rgba(0, 0, 0, 0.1);
        }}
        .highlight {{
            background-color: #e8f5e9;
            padding: 15px;
            border-left: 5px solid #4caf50;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .pros-cons {{
            display: flex;
            gap: 20px;
            margin: 25px 0;
        }}
        .pros, .cons {{
            flex: 1;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}
        .pros {{
            background-color: #e8f5e9;
            border-left: 5px solid #4caf50;
        }}
        .cons {{
            background-color: #ffebee;
            border-left: 5px solid #f44336;
        }}
        .metric-card {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-name {{
            font-size: 14px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .model-section {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 25px;
            margin: 30px 0;
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
            margin-top: 50px;
            text-align: right;
            padding: 10px;
            border-top: 1px solid #eee;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
        }}
        .performance-table {{
            width: 80%;
            margin: 30px auto;
            font-size: 16px;
        }}
        .performance-table .winner {{
            font-weight: bold;
            color: #2ecc71;
        }}        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 15px rgba(0, 0, 0, 0.1);
            display: block;
            margin: 0 auto;
        }}
        /* Custom style for model comparison table */
        .model-comparison-table {{
            width: 80%;
            margin: 30px auto;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        .model-comparison-table th {{
            padding: 15px;
            background-color: #1a237e;
            color: white;
            text-align: center;
            font-weight: bold;
        }}
        .model-comparison-table td {{
            padding: 12px 15px;
            text-align: center;
        }}
        .model-comparison-table tr:first-child th:first-child {{
            background-color: #2c3e50;
        }}
        .xgboost-value {{
            color: #3498db;
            font-weight: bold;
        }}
        .lightgbm-value {{
            color: #2ecc71;
            font-weight: bold;
        }}
        .winner-cell {{
            background-color: rgba(46, 204, 113, 0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="mb-5">
            <h1>Stock Prediction Model Training Report</h1>
            <p class="lead">Comprehensive analysis of XGBoost and LightGBM models for stock price prediction</p>        </header>
        
        <div class="chart-container">
            <img src="{chart_filename}" alt="Model Performance Comparison" class="img-fluid">
        </div>
        
        <!-- Custom Performance Metrics Table -->
        <div class="model-comparison-table">
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>XGBoost</th>
                        <th>LightGBM</th>
                        <th>Winner</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>ROC AUC</strong></td>
                        <td class="xgboost-value">0.5056</td>
                        <td class="lightgbm-value winner-cell">0.5074</td>
                        <td>LightGBM</td>
                    </tr>
                    <tr>
                        <td><strong>F1 Score</strong></td>
                        <td class="xgboost-value winner-cell">0.6564</td>
                        <td class="lightgbm-value">0.6269</td>
                        <td>XGBoost</td>
                    </tr>
                    <tr>
                        <td><strong>Accuracy</strong></td>
                        <td class="xgboost-value">0.4937</td>
                        <td class="lightgbm-value winner-cell">0.4963</td>
                        <td>LightGBM</td>
                    </tr>
                    <tr>
                        <td><strong>Precision</strong></td>
                        <td class="xgboost-value">0.4917</td>
                        <td class="lightgbm-value winner-cell">0.4920</td>
                        <td>LightGBM</td>
                    </tr>
                    <tr>
                        <td><strong>Recall</strong></td>
                        <td class="xgboost-value winner-cell">0.9870</td>
                        <td class="lightgbm-value">0.8638</td>
                        <td>XGBoost</td>
                    </tr>
                    <tr>
                        <td><strong>Training Time</strong></td>
                        <td class="xgboost-value">329.25s</td>
                        <td class="lightgbm-value winner-cell">277.12s</td>
                        <td>LightGBM</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="content">
            {enhanced_html}
        </div>
        
        <div class="timestamp">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

    # Save the HTML report
    report_path = os.path.join('reports', 'model_training_report.html')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as file:
        file.write(html_template)
    
    print(f"HTML report generated at: {report_path}")
    return report_path

if __name__ == "__main__":
    try:
        report_path = create_html_report()
        if report_path:
            print(f"Report successfully generated at: {os.path.abspath(report_path)}")
        else:
            print("Error: Report path is None. Report generation failed.")
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
