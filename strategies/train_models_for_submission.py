"""
Pre-train models for competition submission
Run this ONCE before submitting code to teaching team

Trains on data up to Feb 27, 2026 and saves model files
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from data.data_fetcher import fetch_asset_data
from data.data_pipeline import calculate_basic_features, create_cross_asset_features, prepare_ml_features
from data.constants import ASSETS
from data.model import PortfolioRiskOptimizer, RegimeFilter


def train_and_save_models(training_end_date='2026-02-21', save_dir='models'):
    """
    Train models on historical data and save to disk
    
    Args:
        training_end_date: Last date of training data (day before competition starts)
        save_dir: Directory to save model files
    """
    
    print("="*60)
    print("PRE-TRAINING MODELS FOR COMPETITION")
    print("="*60)
    print(f"Training end date: {training_end_date}")
    print(f"Save directory: {save_dir}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Fetch training data
    print("\n📥 Fetching training data...")
    start_date = pd.to_datetime(training_end_date) - pd.Timedelta(days=782)  # 2 years
    end_date = pd.to_datetime(training_end_date)
    
    raw_data = fetch_asset_data(
        symbol_mapping=ASSETS,
        is_backtesting=True,
        start_date=start_date,
        end_date=end_date
    )
    
    if not raw_data:
        raise ValueError("No training data fetched!")
    
    # Feature engineering
    print("\n🔧 Engineering features...")
    featured = {sym: calculate_basic_features(df) for sym, df in raw_data.items()}
    enhanced = create_cross_asset_features(featured)
    ml_df = prepare_ml_features(enhanced)
    
    if ml_df.empty:
        raise ValueError("No ML features created!")
    
    print(f"✅ Created features: {ml_df.shape}")
    print(f"   Date range: {ml_df.index.min().date()} to {ml_df.index.max().date()}")
    
    # Prepare X, y
    X = ml_df.drop(columns=["symbol", "target_return", "target_volatility"])
    y_ret = ml_df["target_return"]
    y_vol = ml_df["target_volatility"]
    
    # Clean
    mask = np.isfinite(y_ret.values) & np.isfinite(y_vol.values)
    X_clean = X.loc[mask]
    y_ret_clean = y_ret.loc[mask]
    y_vol_clean = y_vol.loc[mask]
    
    print(f"\n🧹 Cleaned data: {len(X_clean)} samples")
    
    # Train optimizer (this trains all 3 models)
    print("\n🤖 Training models...")
    optimizer = PortfolioRiskOptimizer(
        risk_target=0.15,
        direction_gate_threshold=0.0
    )
    
    optimizer.fit(X_clean, y_ret_clean, y_vol_clean, validation_split=0.2)
    
    # Save optimizer (includes all 3 models + scaler)
    model_path = os.path.join(save_dir, 'portfolio_optimizer.pkl')
    optimizer.save(model_path)
    print(f"\n💾 Saved optimizer to: {model_path}")
    
    # Save training metadata
    metadata = {
        'training_end_date': training_end_date,
        'training_start_date': start_date.strftime('%Y-%m-%d'),
        'n_samples': len(X_clean),
        'features': list(X_clean.columns),
        'symbols': list(raw_data.keys()),
        'val_metrics': optimizer.last_val_metrics,
        'created_at': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(save_dir, 'training_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata to: {metadata_path}")
    
    # Verify models work
    print("\n✅ Verification:")
    print(f"   Vol R²: {metadata['val_metrics']['vol_r2']:.3f}")
    print(f"   Dir Acc: {metadata['val_metrics']['dir_acc']:.3f}")
    print(f"   Dir AUC: {metadata['val_metrics']['dir_auc']:.3f}")
    
    print("\n" + "="*60)
    print("✅ PRE-TRAINING COMPLETE")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - {model_path}")
    print(f"  - {metadata_path}")
    print("\nNext steps:")
    print("  1. Update strategy to use load_pretrained=True")
    print("  2. Include models/ folder in submission")
    print("  3. Test with: python run_competition_backtest.py")


if __name__ == "__main__":
    train_and_save_models()