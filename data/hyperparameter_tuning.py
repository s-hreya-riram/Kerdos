"""
Hyperparameter Tuning for XGBoost Portfolio Models

This script finds optimal XGBoost parameters for:
1. Volatility prediction model (high priority - R² = 0.91)
2. Return prediction model (lower priority - R² = -0.14)

Approach: Time-series cross-validation with walk-forward splits
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from itertools import product
import json
from datetime import datetime

# Add your project paths
import os
import sys
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from data.data_fetcher import fetch_asset_data
from data.data_pipeline import calculate_basic_features, create_cross_asset_features, prepare_ml_features
from data.constants import ASSETS


def load_training_data(start_date='2024-01-01', end_date='2026-02-06'):
    """
    Load and prepare data for hyperparameter tuning
    """
    print("="*60)
    print("LOADING TRAINING DATA")
    print("="*60)
    
    # Fetch data
    raw_data = fetch_asset_data(
        symbol_mapping=ASSETS,
        is_backtesting=True,
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date)
    )
    
    # Feature engineering
    featured = {}
    for sym, df in raw_data.items():
        featured[sym] = calculate_basic_features(df)
    
    enhanced = create_cross_asset_features(featured)
    ml_df = prepare_ml_features(enhanced)
    
    # Prepare X, y
    X = ml_df.drop(columns=["symbol", "target_return", "target_volatility"])
    y_ret = ml_df["target_return"]
    y_vol = ml_df["target_volatility"]
    
    # Remove NaN/inf values
    mask = np.isfinite(y_ret.values) & np.isfinite(y_vol.values)
    X_clean = X.loc[mask]
    y_ret_clean = y_ret.loc[mask]
    y_vol_clean = y_vol.loc[mask]
    
    print(f"✅ Loaded {len(X_clean)} clean samples")
    print(f"   Date range: {X_clean.index.min().date()} to {X_clean.index.max().date()}")
    print(f"   Features: {X_clean.shape[1]}")
    
    return X_clean, y_ret_clean, y_vol_clean


def time_series_cv_score(X, y, params, n_splits=5, model_type='vol'):
    """
    Time-series cross-validation with walk-forward splits
    
    Returns: Mean R² across all folds
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        
        # Predict and score
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        scores.append(r2)
    
    return np.mean(scores)


def grid_search_xgb(X, y, param_grid, model_type='vol', n_splits=3):
    """
    Grid search over hyperparameters
    
    Args:
        X: Features
        y: Targets (returns or volatility)
        param_grid: Dictionary of parameter ranges
        model_type: 'vol' or 'ret' (for logging)
        n_splits: Number of CV splits (fewer = faster)
    
    Returns:
        best_params, best_score, all_results
    """
    print(f"\n{'='*60}")
    print(f"GRID SEARCH FOR {model_type.upper()} MODEL")
    print(f"{'='*60}")
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    print(f"Testing {len(param_combinations)} parameter combinations")
    print(f"Using {n_splits}-fold time-series CV")
    print(f"Estimated time: ~{len(param_combinations) * n_splits * 2} seconds\n")
    
    results = []
    best_score = -np.inf
    best_params = None
    
    for i, combo in enumerate(param_combinations):
        # Create params dict
        params = dict(zip(param_names, combo))
        params['objective'] = 'reg:squarederror'
        params['random_state'] = 42
        params['n_jobs'] = -1
        
        # Score this combination
        try:
            score = time_series_cv_score(X, y, params, n_splits=n_splits)
            
            results.append({
                'params': params,
                'mean_r2': score
            })
            
            # Track best
            if score > best_score:
                best_score = score
                best_params = params
                print(f"✅ New best! R² = {score:.4f}")
                print(f"   Params: {params}")
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{len(param_combinations)} ({(i+1)/len(param_combinations)*100:.1f}%)")
        
        except Exception as e:
            print(f"   ❌ Error with params {params}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best R²: {best_score:.4f}")
    print(f"Best params: {best_params}")
    
    return best_params, best_score, results


def random_search_xgb(X, y, param_distributions, model_type='vol', n_iterations=50, n_splits=3):
    """
    Random search over hyperparameters (faster than grid search)
    
    Args:
        X: Features
        y: Targets
        param_distributions: Dict with (min, max) for each param
        model_type: 'vol' or 'ret'
        n_iterations: Number of random combinations to try
        n_splits: Number of CV splits
    
    Returns:
        best_params, best_score, all_results
    """
    print(f"\n{'='*60}")
    print(f"RANDOM SEARCH FOR {model_type.upper()} MODEL")
    print(f"{'='*60}")
    print(f"Testing {n_iterations} random parameter combinations")
    print(f"Using {n_splits}-fold time-series CV")
    print(f"Estimated time: ~{n_iterations * n_splits * 2} seconds\n")
    
    results = []
    best_score = -np.inf
    best_params = None
    
    np.random.seed(42)
    
    for i in range(n_iterations):
        # Sample random parameters
        params = {}
        for param_name, (min_val, max_val, param_type) in param_distributions.items():
            if param_type == 'int':
                params[param_name] = np.random.randint(min_val, max_val + 1)
            elif param_type == 'float':
                params[param_name] = np.random.uniform(min_val, max_val)
            elif param_type == 'log':
                # Log-uniform distribution (good for learning_rate)
                params[param_name] = 10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))
        
        params['objective'] = 'reg:squarederror'
        params['random_state'] = 42
        params['n_jobs'] = -1
        
        # Score this combination
        try:
            score = time_series_cv_score(X, y, params, n_splits=n_splits)
            
            results.append({
                'params': params,
                'mean_r2': score
            })
            
            # Track best
            if score > best_score:
                best_score = score
                best_params = params
                print(f"✅ New best! R² = {score:.4f}")
                print(f"   Params: {params}")
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i+1}/{n_iterations} ({(i+1)/n_iterations*100:.1f}%)")
        
        except Exception as e:
            print(f"   ❌ Error with params {params}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"RANDOM SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best R²: {best_score:.4f}")
    print(f"Best params: {best_params}")
    
    return best_params, best_score, results


def bayesian_search_xgb(X, y, param_space, model_type='vol', n_iterations=50, n_splits=3):
    """
    Bayesian optimization (requires optuna)
    More efficient than random search
    
    Install: pip install optuna
    """
    try:
        import optuna
    except ImportError:
        print("⚠️  Optuna not installed. Run: pip install optuna")
        return None, None, None
    
    print(f"\n{'='*60}")
    print(f"BAYESIAN OPTIMIZATION FOR {model_type.upper()} MODEL")
    print(f"{'='*60}")
    
    def objective(trial):
        # Sample parameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *param_space['n_estimators']),
            'max_depth': trial.suggest_int('max_depth', *param_space['max_depth']),
            'learning_rate': trial.suggest_float('learning_rate', *param_space['learning_rate'], log=True),
            'subsample': trial.suggest_float('subsample', *param_space['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *param_space['colsample_bytree']),
            'min_child_weight': trial.suggest_int('min_child_weight', *param_space.get('min_child_weight', (1, 10))),
            'gamma': trial.suggest_float('gamma', *param_space.get('gamma', (0, 0.5))),
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Score
        score = time_series_cv_score(X, y, params, n_splits=n_splits)
        return score
    
    # Create study
    study = optuna.create_study(direction='maximize', study_name=f'{model_type}_optimization')
    study.optimize(objective, n_trials=n_iterations, show_progress_bar=True)
    
    print(f"\n{'='*60}")
    print(f"BAYESIAN OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best R²: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Convert to full params dict
    best_params = study.best_params.copy()
    best_params['objective'] = 'reg:squarederror'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    return best_params, study.best_value, study.trials


def save_results(vol_params, vol_score, ret_params, ret_score, results_dir='hyperparameter_tuning'):
    """
    Save tuning results to file
    """
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'timestamp': timestamp,
        'volatility_model': {
            'best_params': vol_params,
            'best_r2': float(vol_score)
        },
        'return_model': {
            'best_params': ret_params,
            'best_r2': float(ret_score)
        }
    }
    
    # Save as JSON
    filepath = os.path.join(results_dir, f'best_params_{timestamp}.json')
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {filepath}")
    
    # Also save as Python constants file
    py_filepath = os.path.join(results_dir, f'best_params_{timestamp}.py')
    with open(py_filepath, 'w') as f:
        f.write(f"# Auto-generated hyperparameters - {timestamp}\n\n")
        f.write(f"VOL_MODEL_PARAMS = {vol_params}\n\n")
        f.write(f"RET_MODEL_PARAMS = {ret_params}\n\n")
        f.write(f"# Validation scores:\n")
        f.write(f"# Volatility R²: {vol_score:.4f}\n")
        f.write(f"# Return R²: {ret_score:.4f}\n")
    
    print(f"✅ Python file saved to: {py_filepath}")


# ========== MAIN TUNING WORKFLOW ==========

def main():
    """
    Main hyperparameter tuning workflow
    """
    print("="*60)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    
    # Load data
    X, y_ret, y_vol = load_training_data()

    # ========== APPROACH 1: COARSE GRID SEARCH (FAST) ==========
    print("\n" + "="*60)
    print("APPROACH 1: COARSE GRID SEARCH")
    print("="*60)
    
    # Coarse grid for volatility model (most important)
    vol_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    vol_params_grid, vol_score_grid, vol_results_grid = grid_search_xgb(
        X, y_vol, vol_grid, model_type='vol', n_splits=3
    )
    
    # ========== APPROACH 2: RANDOM SEARCH (BETTER) ==========
    print("\n" + "="*60)
    print("APPROACH 2: RANDOM SEARCH")
    print("="*60)
    
    # Random search distributions for volatility
    vol_distributions = {
        'n_estimators': (100, 500, 'int'),
        'max_depth': (2, 8, 'int'),
        'learning_rate': (0.001, 0.3, 'log'),
        'subsample': (0.5, 1.0, 'float'),
        'colsample_bytree': (0.5, 1.0, 'float'),
        'min_child_weight': (1, 10, 'int'),
        'gamma': (0, 0.5, 'float')
    }
    
    vol_params_random, vol_score_random, vol_results_random = random_search_xgb(
        X, y_vol, vol_distributions, model_type='vol', n_iterations=50, n_splits=3
    )
    
    # ========== APPROACH 3: BAYESIAN OPTIMIZATION (BEST) ==========
    print("\n" + "="*60)
    print("APPROACH 3: BAYESIAN OPTIMIZATION (if optuna installed)")
    print("="*60)
    
    # Bayesian search space for volatility
    vol_space = {
        'n_estimators': (100, 500),
        'max_depth': (2, 8),
        'learning_rate': (0.001, 0.3),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'min_child_weight': (1, 10),
        'gamma': (0, 0.5)
    }
    
    vol_params_bayes, vol_score_bayes, vol_trials_bayes = bayesian_search_xgb(
        X, y_vol, vol_space, model_type='vol', n_iterations=50, n_splits=3
    )
    
    # ========== SELECT BEST APPROACH ==========
    best_vol_params = vol_params_bayes if vol_params_bayes else vol_params_random
    best_vol_score = vol_score_bayes if vol_params_bayes else vol_score_random
    
    # ========== TUNE RETURN MODEL (Optional - it's performing poorly) ==========
    print("\n" + "="*60)
    print("TUNING RETURN MODEL (Quick)")
    print("="*60)
    print("Note: Return model R² is negative, so this may not help much")
    
    # Quick random search for returns
    ret_params, ret_score, ret_results = random_search_xgb(
        X, y_ret, vol_distributions, model_type='ret', n_iterations=20, n_splits=3
    )
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*60)
    print(f"\n📊 VOLATILITY MODEL:")
    print(f"   Best R²: {best_vol_score:.4f}")
    print(f"   Current R²: 0.687 (baseline)")
    print(f"   Improvement: {(best_vol_score - 0.687) * 100:.1f}%")
    print(f"   Best params: {best_vol_params}")
    
    print(f"\n📊 RETURN MODEL:")
    print(f"   Best R²: {ret_score:.4f}")
    print(f"   Current R²: -0.141 (baseline)")
    print(f"   Improvement: {(ret_score - (-0.141)) * 100:.1f}%")
    print(f"   Best params: {ret_params}")
    
    # Save results
    save_results(best_vol_params, best_vol_score, ret_params, ret_score)
    
    print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    main()