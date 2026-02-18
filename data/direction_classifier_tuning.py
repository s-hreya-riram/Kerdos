"""
Direction Classifier Hyperparameter Tuning

Separate tuning for the XGBoost direction classifier since binary classification
has different overfitting characteristics than regression.

Focus: Maximize AUC on held-out validation set
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import os
import sys

# Add project paths
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from data.data_fetcher import fetch_asset_data
from data.data_pipeline import calculate_basic_features, create_cross_asset_features, prepare_ml_features
from data.constants import ASSETS

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️  Optuna not installed. Run: pip install optuna")
    OPTUNA_AVAILABLE = False


def load_training_data(start_date='2024-01-01', end_date='2026-02-18'):
    """Load and prepare data for hyperparameter tuning"""
    print("="*60)
    print("LOADING TRAINING DATA")
    print("="*60)
    
    raw_data = fetch_asset_data(
        symbol_mapping=ASSETS,
        is_backtesting=True,
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date)
    )
    
    featured = {sym: calculate_basic_features(df) for sym, df in raw_data.items()}
    enhanced = create_cross_asset_features(featured)
    ml_df = prepare_ml_features(enhanced)
    
    X = ml_df.drop(columns=["symbol", "target_return", "target_volatility"])
    y_ret = ml_df["target_return"]
    
    # Clean
    mask = np.isfinite(y_ret.values)
    X_clean = X.loc[mask]
    y_ret_clean = y_ret.loc[mask]
    
    # Direction target
    y_dir = (y_ret_clean > 0).astype(int)
    
    print(f"✅ Loaded {len(X_clean)} clean samples")
    print(f"   Date range: {X_clean.index.min().date()} to {X_clean.index.max().date()}")
    print(f"   Features: {X_clean.shape[1]}")
    print(f"   Direction balance: {y_dir.mean():.1%} up / {1-y_dir.mean():.1%} down")
    
    return X_clean, y_dir


def temporal_train_test_split(X, y, validation_split=0.2):
    """Split by time, not random sampling"""
    unique_dates = X.index.unique().sort_values()
    split_idx = int(len(unique_dates) * (1 - validation_split))
    
    train_end_date = unique_dates[split_idx - 1]
    val_start_date = unique_dates[split_idx]
    
    train_mask = X.index <= train_end_date
    val_mask = X.index >= val_start_date
    
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    
    print(f"\n📊 Train/Val Split:")
    print(f"   Train: {len(X_train)} rows ({X_train.index.min().date()} → {X_train.index.max().date()})")
    print(f"   Val:   {len(X_val)} rows ({X_val.index.min().date()} → {X_val.index.max().date()})")
    print(f"   Gap: {(X_val.index.min() - X_train.index.max()).days}d")
    
    return X_train, X_val, y_train, y_val


def objective(trial, X_train, X_val, y_train, y_val):
    """Optuna objective function - maximize AUC"""
    
    # Sample hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.5),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
    }
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    # Train
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_s, y_train, verbose=False)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_val_s)[:, 1]
    
    # Score on AUC (this is what we want to maximize)
    try:
        auc = roc_auc_score(y_val, y_pred_proba)
    except ValueError:
        # If only one class in val set (shouldn't happen with temporal split)
        auc = 0.5
    
    # Also log accuracy for reference
    y_pred = (y_pred_proba >= 0.5).astype(int)
    acc = accuracy_score(y_val, y_pred)
    
    # Store both as user attributes
    trial.set_user_attr("accuracy", acc)
    
    return auc  # Optuna will maximize this


def run_bayesian_optimization(X_train, X_val, y_train, y_val, n_trials=100):
    """Run Optuna Bayesian optimization"""
    
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna not available. Install with: pip install optuna")
        return None
    
    print(f"\n{'='*60}")
    print(f"BAYESIAN OPTIMIZATION (Optuna)")
    print(f"{'='*60}")
    print(f"Running {n_trials} trials to maximize AUC")
    print(f"This will take approximately {n_trials * 2 // 60} minutes\n")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='direction_classifier_tuning',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n🏆 Best Trial:")
    print(f"   AUC: {study.best_value:.4f}")
    print(f"   Accuracy: {study.best_trial.user_attrs['accuracy']:.4f}")
    print(f"\n📋 Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    return study


def validate_best_params(X_train, X_val, y_train, y_val, params):
    """Validate the best parameters with full metrics"""
    
    print(f"\n{'='*60}")
    print("VALIDATION ON BEST PARAMS")
    print(f"{'='*60}")
    
    # Add fixed params
    full_params = {
        **params,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
    }
    
    # Train final model
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    model = xgb.XGBClassifier(**full_params)
    model.fit(X_train_s, y_train, verbose=False)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_val_s)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_val, y_pred_proba)
    acc = accuracy_score(y_val, y_pred)
    
    print(f"\n📊 Validation Metrics:")
    print(f"   AUC: {auc:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    
    # Compare to baseline (always predict majority class)
    baseline_pred = np.ones(len(y_val)) if y_val.mean() > 0.5 else np.zeros(len(y_val))
    baseline_acc = accuracy_score(y_val, baseline_pred)
    print(f"   Baseline Acc (majority): {baseline_acc:.4f}")
    print(f"   Improvement: {(acc - baseline_acc):.4f}")
    
    return auc, acc


def save_results(params, auc, acc, output_dir='hyperparameter_tuning'):
    """Save tuning results to file"""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'timestamp': timestamp,
        'direction_classifier': {
            'best_params': params,
            'val_auc': float(auc),
            'val_accuracy': float(acc),
        }
    }
    
    # JSON
    json_path = os.path.join(output_dir, f'direction_classifier_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {json_path}")
    
    # Python constants file
    py_path = os.path.join(output_dir, f'direction_classifier_{timestamp}.py')
    with open(py_path, 'w') as f:
        f.write(f"# Direction Classifier Hyperparameters - {timestamp}\n\n")
        f.write(f"XGB_CLASSIFIER_PARAMS = {params}\n\n")
        f.write(f"# Validation Metrics:\n")
        f.write(f"# AUC: {auc:.4f}\n")
        f.write(f"# Accuracy: {acc:.4f}\n")
    
    print(f"✅ Python file saved to: {py_path}")
    
    # Instructions
    print(f"\n📝 To use these params, update data/constants.py:")
    print(f"   Replace XGB_CLASSIFIER_PARAMS with the params from {py_path}")


def main():
    """Main hyperparameter tuning workflow"""
    
    print("="*60)
    print("DIRECTION CLASSIFIER HYPERPARAMETER TUNING")
    print("="*60)
    print(f"Start time: {datetime.now()}\n")
    
    # Load data
    X, y_dir = load_training_data()
    
    # Split
    X_train, X_val, y_train, y_val = temporal_train_test_split(X, y_dir)
    
    # Run Bayesian optimization
    study = run_bayesian_optimization(X_train, X_val, y_train, y_val, n_trials=100)
    
    if study is None:
        return
    
    # Validate best params
    auc, acc = validate_best_params(X_train, X_val, y_train, y_val, study.best_params)
    
    # Save results
    save_results(study.best_params, auc, acc)
    
    print(f"\nEnd time: {datetime.now()}")
    print("="*60)


if __name__ == "__main__":
    main()