"""
Advanced Model Training Module

This module provides comprehensive model selection, training, and evaluation
with multiple algorithms and hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import joblib
import os
from datetime import datetime

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Preprocessing and evaluation
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

# Optimization
from scipy.stats import randint, uniform
import optuna

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedModelTrainer:
    """Advanced model training with multiple algorithms and optimization."""
    
    def __init__(self, model_dir: str = "models", random_state: int = 42):
        self.model_dir = model_dir
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.best_model = None
        self.results = {}
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations and hyperparameter spaces."""
        return {
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'param_space': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'bootstrap': [True, False]
                }
            },
            'xgboost': {
                'model': XGBClassifier(random_state=self.random_state, n_jobs=-1, use_label_encoder=False),
                'param_space': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.3, 0.5],
                    'reg_alpha': [0, 0.1, 0.5, 1],
                    'reg_lambda': [1, 1.5, 2, 3]
                }
            },
            'lightgbm': {
                'model': LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1),
                'param_space': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, -1],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'num_leaves': [31, 50, 100, 200],
                    'feature_fraction': [0.5, 0.7, 0.9],
                    'bagging_fraction': [0.5, 0.7, 0.9],
                    'bagging_freq': [1, 3, 5],
                    'min_child_samples': [20, 50, 100]
                }
            },
            'catboost': {
                'model': CatBoostClassifier(random_state=self.random_state, verbose=False),
                'param_space': {
                    'iterations': [100, 200, 300],
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'border_count': [32, 64, 128, 255]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'param_space': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier(random_state=self.random_state, n_jobs=-1),
                'param_space': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'param_space': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'param_space': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                }
            }
        }
    
    def preprocess_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          scaler_type: str = 'standard',
                          feature_selection: Optional[str] = None,
                          n_features: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features with scaling and optional feature selection.
        
        Args:
            X_train: Training features
            X_test: Test features
            scaler_type: Type of scaler ('standard', 'robust')
            feature_selection: Feature selection method ('f_classif', 'mutual_info')
            n_features: Number of features to select
            
        Returns:
            Processed X_train, X_test
        """
        # Scaling
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        if feature_selection and n_features:
            if feature_selection == 'f_classif':
                selector = SelectKBest(f_classif, k=n_features)
            elif feature_selection == 'mutual_info':
                selector = SelectKBest(mutual_info_classif, k=n_features)
            else:
                raise ValueError(f"Unknown feature selection: {feature_selection}")
            
            X_train_scaled = selector.fit_transform(X_train_scaled, y_train)
            X_test_scaled = selector.transform(X_test_scaled)
            
            # Store selected features
            selected_features = X_train.columns[selector.get_support()]
            self.feature_selectors[f"{feature_selection}_{n_features}"] = {
                'selector': selector,
                'features': selected_features.tolist()
            }
        
        self.scalers[scaler_type] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          optimization: str = 'grid', n_iter: int = 50) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter optimization.
        
        Args:
            model_name: Name of the model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            optimization: Type of optimization ('grid', 'random', 'optuna')
            n_iter: Number of iterations for optimization
            
        Returns:
            Training results
        """
        print(f"\nTraining {model_name}...")
        
        config = self.get_model_configs()[model_name]
        model = config['model']
        param_space = config['param_space']
        
        # Hyperparameter optimization
        if optimization == 'grid':
            search = GridSearchCV(model, param_space, cv=3, scoring='roc_auc', n_jobs=-1)
        elif optimization == 'random':
            search = RandomizedSearchCV(model, param_space, n_iter=n_iter, cv=3, 
                                      scoring='roc_auc', n_jobs=-1, random_state=self.random_state)
        elif optimization == 'optuna':
            # Use Optuna for more advanced optimization
            best_params = self._optuna_optimization(model_name, X_train, y_train, n_trials=n_iter)
            model.set_params(**best_params)
            search = model
        else:
            search = model
        
        # Train
        if optimization in ['grid', 'random']:
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
        else:
            best_model = search.fit(X_train, y_train)
            best_params = best_model.get_params() if hasattr(best_model, 'get_params') else {}
        
        # Evaluate
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        
        results = {
            'model': best_model,
            'best_params': best_params,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{model_name} - AUC: {results['roc_auc']:.4f}, F1: {results['f1']:.4f}")
        
        return results
    
    def _optuna_optimization(self, model_name: str, X_train: np.ndarray, 
                           y_train: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
        """Use Optuna for hyperparameter optimization."""
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1, 3)
                }
                model = XGBClassifier(**params, random_state=self.random_state, n_jobs=-1)
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 200),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100)
                }
                model = LGBMClassifier(**params, random_state=self.random_state, n_jobs=-1)
            else:
                return 0
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      model_names: List[str]) -> Dict[str, Any]:
        """
        Train an ensemble of models.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_names: List of models to include in ensemble
            
        Returns:
            Ensemble results
        """
        from sklearn.ensemble import VotingClassifier
        
        # Train individual models
        estimators = []
        for name in model_names:
            result = self.train_single_model(name, X_train, y_train, X_val, y_val)
            estimators.append((name, result['model']))
            self.models[name] = result
        
        # Create ensemble
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred = ensemble.predict(X_val)
        y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
        
        ensemble_results = {
            'model': ensemble,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"\nEnsemble - AUC: {ensemble_results['roc_auc']:.4f}, F1: {ensemble_results['f1']:.4f}")
        
        return ensemble_results
    
    def calibrate_model(self, model: Any, X_cal: np.ndarray, y_cal: np.ndarray,
                       method: str = 'isotonic') -> Any:
        """
        Calibrate model probabilities.
        
        Args:
            model: Trained model
            X_cal, y_cal: Calibration data
            method: Calibration method ('isotonic' or 'sigmoid')
            
        Returns:
            Calibrated model
        """
        calibrated = CalibratedClassifierCV(model, method=method, cv='prefit')
        calibrated.fit(X_cal, y_cal)
        return calibrated
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all trained models on test set.
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Evaluation results DataFrame
        """
        results = []
        
        for name, model_result in self.models.items():
            model = model_result['model']
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results.append({
                'model': name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            })
        
        results_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
        
        # Identify best model
        self.best_model = self.models[results_df.iloc[0]['model']]
        
        return results_df
    
    def plot_model_comparison(self, results_df: pd.DataFrame):
        """Plot model performance comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        results_df.set_index('model')[metrics].plot(kind='bar', ax=axes[0])
        axes[0].set_title('Model Performance Metrics')
        axes[0].set_ylabel('Score')
        axes[0].legend(loc='lower right')
        axes[0].set_ylim(0, 1)
        
        # ROC AUC focus
        results_df.sort_values('roc_auc', ascending=True).plot.barh(
            x='model', y='roc_auc', ax=axes[1], legend=False
        )
        axes[1].set_title('Model ROC AUC Scores')
        axes[1].set_xlabel('ROC AUC')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """Save all trained models and preprocessing objects."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save models
        for name, model_result in self.models.items():
            model_path = os.path.join(self.model_dir, f'{name}_{timestamp}.pkl')
            joblib.dump(model_result['model'], model_path)
            print(f"Saved {name} to {model_path}")
        
        # Save best model
        if self.best_model:
            best_path = os.path.join(self.model_dir, f'best_model_{timestamp}.pkl')
            joblib.dump(self.best_model['model'], best_path)
            print(f"Saved best model to {best_path}")
        
        # Save preprocessing objects
        preprocessing = {
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors
        }
        preprocess_path = os.path.join(self.model_dir, f'preprocessing_{timestamp}.pkl')
        joblib.dump(preprocessing, preprocess_path)
        print(f"Saved preprocessing to {preprocess_path}")
        
        # Save results
        results_path = os.path.join(self.model_dir, f'results_{timestamp}.pkl')
        joblib.dump(self.results, results_path)
    
    def load_model(self, model_path: str):
        """Load a saved model."""
        return joblib.load(model_path)
    
    def generate_training_report(self, results_df: pd.DataFrame):
        """Generate comprehensive training report."""
        report = []
        report.append("# Model Training Report\n")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        
        report.append("\n## Model Performance Summary\n")
        report.append(results_df.to_string())
        
        report.append("\n\n## Best Model\n")
        if self.best_model:
            best_name = results_df.iloc[0]['model']
            report.append(f"Model: {best_name}\n")
            report.append(f"Parameters: {self.best_model.get('best_params', {})}\n")
        
        report.append("\n## Feature Selection\n")
        for name, selector_info in self.feature_selectors.items():
            report.append(f"\n### {name}\n")
            report.append(f"Selected features: {', '.join(selector_info['features'][:10])}...\n")
        
        # Save report
        report_path = os.path.join(self.model_dir, 'training_report.md')
        with open(report_path, 'w') as f:
            f.writelines(report)
        
        print(f"\nTraining report saved to {report_path}") 