#!/usr/bin/env python3
"""
Feature Engineering EDA and Analysis

This script performs comprehensive analysis of features for stock prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
sys.path.append('..')
from src.data.data_loader import StockDataLoader
from src.data.feature_engineering import FeatureGenerator

def analyze_feature_distributions(df, numeric_features):
    """Analyze and visualize feature distributions."""
    print("\n=== FEATURE DISTRIBUTION ANALYSIS ===")
    
    # Calculate statistics
    stats_df = pd.DataFrame({
        'mean': df[numeric_features].mean(),
        'std': df[numeric_features].std(),
        'skew': df[numeric_features].skew(),
        'kurtosis': df[numeric_features].kurtosis(),
        'missing_pct': (df[numeric_features].isnull().sum() / len(df)) * 100
    })
    
    print("\nFeature Statistics Summary:")
    print(stats_df.round(3).head(20))
    
    # Plot distributions
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.ravel()
    
    for idx, feature in enumerate(numeric_features[:16]):
        data = df[feature].dropna()
        axes[idx].hist(data, bins=50, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{feature}\nSkew: {stats.skew(data):.2f}')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('../reports/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats_df

def analyze_feature_correlations(df, numeric_features):
    """Analyze feature correlations and multicollinearity."""
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_features].corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    print(f"\nFound {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8):")
    for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
        print(f"  {feat1} <-> {feat2}: {corr:.3f}")
    
    # Visualize correlation matrix
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('../reports/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return high_corr_pairs

def analyze_feature_importance(df, numeric_features):
    """Analyze feature importance using multiple methods."""
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Prepare data
    df_clean = df.dropna()
    X = df_clean[numeric_features]
    y = df_clean['Label']
    
    # 1. Random Forest Feature Importance
    print("\nTraining Random Forest for feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 2. Mutual Information
    print("Calculating Mutual Information scores...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_importance = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # 3. F-statistic
    print("Calculating F-statistic scores...")
    f_scores, _ = f_classif(X, y)
    f_importance = pd.DataFrame({
        'feature': X.columns,
        'f_score': f_scores
    }).sort_values('f_score', ascending=False)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # RF importance
    axes[0].barh(rf_importance['feature'][:15], rf_importance['importance'][:15])
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Random Forest Feature Importance (Top 15)')
    axes[0].invert_yaxis()
    
    # MI scores
    axes[1].barh(mi_importance['feature'][:15], mi_importance['mi_score'][:15])
    axes[1].set_xlabel('MI Score')
    axes[1].set_title('Mutual Information Score (Top 15)')
    axes[1].invert_yaxis()
    
    # F-scores
    axes[2].barh(f_importance['feature'][:15], f_importance['f_score'][:15])
    axes[2].set_xlabel('F Score')
    axes[2].set_title('F-Statistic Score (Top 15)')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('../reports/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find consensus features
    top_rf = set(rf_importance['feature'][:10])
    top_mi = set(mi_importance['feature'][:10])
    top_f = set(f_importance['feature'][:10])
    consensus = top_rf.intersection(top_mi).intersection(top_f)
    
    print(f"\nConsensus top features (appearing in all methods): {consensus}")
    
    return rf_importance, mi_importance, f_importance, consensus

def analyze_predictive_power(df, numeric_features):
    """Analyze individual feature predictive power."""
    print("\n=== PREDICTIVE POWER ANALYSIS ===")
    
    df_clean = df.dropna()
    results = []
    
    for feature in numeric_features:
        try:
            # Single feature prediction
            X = df_clean[[feature]].values
            y = df_clean['Label'].values
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train simple model
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_scaled, y)
            
            # Calculate AUC
            y_pred_proba = lr.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, y_pred_proba)
            
            results.append({
                'feature': feature,
                'auc': auc,
                'predictive_power': abs(auc - 0.5) * 2
            })
        except:
            pass
    
    # Sort by predictive power
    results_df = pd.DataFrame(results).sort_values('predictive_power', ascending=False)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    plt.barh(results_df['feature'][:20], results_df['predictive_power'][:20])
    plt.xlabel('Predictive Power (0-1)')
    plt.title('Individual Feature Predictive Power (Top 20)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../reports/feature_predictive_power.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTop 10 features by predictive power:")
    print(results_df[['feature', 'auc', 'predictive_power']].head(10))
    
    return results_df

def generate_feature_report(df, numeric_features, high_corr_pairs, rf_importance, consensus):
    """Generate comprehensive feature engineering report."""
    print("\n=== FEATURE ENGINEERING REPORT ===")
    
    report = []
    report.append("# Feature Engineering Analysis Report\n")
    report.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    report.append(f"Total features analyzed: {len(numeric_features)}\n")
    report.append(f"Total samples: {len(df)}\n")
    
    report.append("\n## 1. Top Features by Importance\n")
    report.append("### Random Forest Top 15:\n")
    for idx, row in rf_importance.head(15).iterrows():
        report.append(f"- {row['feature']}: {row['importance']:.4f}\n")
    
    report.append("\n### Consensus Features (appearing in all importance metrics):\n")
    for feat in consensus:
        report.append(f"- {feat}\n")
    
    report.append("\n## 2. Highly Correlated Features to Consider Removing\n")
    for feat1, feat2, corr in high_corr_pairs[:10]:
        report.append(f"- {feat1} and {feat2} (r = {corr:.3f})\n")
    
    report.append("\n## 3. Feature Engineering Recommendations\n")
    report.append("### Features to Add:\n")
    report.append("- **Market Microstructure**: Bid-ask spread, order flow imbalance\n")
    report.append("- **Cross-sectional**: Stock vs sector performance, relative strength\n")
    report.append("- **Alternative Data**: Sentiment scores, news volume\n")
    report.append("- **Macroeconomic**: Interest rates, VIX, dollar index\n")
    report.append("- **Options-based**: Implied volatility, put-call ratio\n")
    
    report.append("\n### Feature Transformations:\n")
    report.append("- Apply log transformation to skewed features\n")
    report.append("- Create interaction terms for top features\n")
    report.append("- Add polynomial features for non-linear relationships\n")
    report.append("- Implement regime-based features (bull/bear market indicators)\n")
    
    # Save report
    with open('../reports/feature_engineering_report.md', 'w') as f:
        f.writelines(report)
    
    print("\nReport saved to reports/feature_engineering_report.md")

def main():
    """Main analysis function."""
    print("=== FEATURE ENGINEERING EDA ===\n")
    
    # Create reports directory
    import os
    os.makedirs('../reports', exist_ok=True)
    
    # Load data
    print("Loading data...")
    data_loader = StockDataLoader()
    stock_data, sp500_data = data_loader.prepare_training_data()
    
    print(f"Stock data shape: {stock_data.shape}")
    print(f"S&P 500 data shape: {sp500_data.shape}")
    
    # Generate features
    print("\nGenerating features...")
    feature_generator = FeatureGenerator()
    df = feature_generator.prepare_features(stock_data, sp500_data)
    
    print(f"Feature dataset shape: {df.shape}")
    
    # Get numeric features, excluding future values (to prevent lookahead bias)
    numeric_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in ['Label', 'Symbol', 'Future_Returns_5d', 'Future_Returns_5d_SP500']]
    
    # Run analyses
    stats_df = analyze_feature_distributions(df, numeric_features)
    high_corr_pairs = analyze_feature_correlations(df, numeric_features)
    rf_imp, mi_imp, f_imp, consensus = analyze_feature_importance(df, numeric_features)
    predictive_df = analyze_predictive_power(df, numeric_features)
    
    # Generate report
    generate_feature_report(df, numeric_features, high_corr_pairs, rf_imp, consensus)
    
    # Save feature recommendations
    import json
    recommendations = {
        'keep_features': list(rf_imp['feature'][:30]),
        'consensus_features': list(consensus),
        'remove_correlated': [pair[0] for pair in high_corr_pairs[:5]],
        'top_predictive': list(predictive_df['feature'][:20])
    }
    
    with open('../data/feature_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Check the reports/ directory for detailed visualizations and analysis.")

if __name__ == "__main__":
    main() 