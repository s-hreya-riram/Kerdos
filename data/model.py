import xgboost as xgb
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from data.constants import XGB_MODEL_PARAMS

class PortfolioRiskOptimizer:
    """
    ML-based portfolio optimizer for time series.
    FIXED: Proper time-series split with no overlap between train and validation
    """

    def __init__(self, risk_target=0.15, model_params=None):
        self.risk_target = risk_target
        self.scaler = StandardScaler()
        self.vol_model = None
        self.ret_model = None
        self.is_trained = False

        self.model_params = model_params or XGB_MODEL_PARAMS
        self.last_val_metrics = None

    
    def fit(self, X, y_ret, y_vol, validation_split=0.2):
        """
        Train with time-series validation split
        
        CRITICAL FIX: The issue was that X might have multiple rows per timestamp
        (one per symbol), so grouping by symbols was creating overlap.
        
        Solution: Split by TEMPORAL INDEX, not by row count
        
        Args:
            X: Features (DataFrame with DatetimeIndex)
            y_ret: Return targets (Series)
            y_vol: Volatility targets (Series)
            validation_split: Fraction of most recent TIME PERIOD for validation
        """
        
        # If too little data, skip validation and train on all
        if len(X) < 100:
            print(f"  ⚠️  Only {len(X)} samples, skipping validation split")
            Xs = self.scaler.fit_transform(X)

            self.vol_model = xgb.XGBRegressor(**self.model_params)
            self.ret_model = xgb.XGBRegressor(**self.model_params)

            self.vol_model.fit(Xs, y_vol)
            self.ret_model.fit(Xs, y_ret)

            self.is_trained = True
            return self
        
        # CRITICAL FIX: Split by TIME, not by row count
        # Get unique timestamps
        unique_dates = X.index.unique().sort_values()
        
        # Split dates into train and validation periods
        split_idx = int(len(unique_dates) * (1 - validation_split))
        train_end_date = unique_dates[split_idx - 1]
        val_start_date = unique_dates[split_idx]
        
        print(f"    🕐 Total unique dates: {len(unique_dates)}")
        print(f"    📅 Train cutoff: {train_end_date.date()}")
        print(f"    📅 Val start: {val_start_date.date()}")
        
        # Split data by date (NO OVERLAP!)
        train_mask = X.index <= train_end_date
        val_mask = X.index >= val_start_date
        
        X_train = X[train_mask]
        X_val = X[val_mask]
        
        y_ret_train = y_ret[train_mask]
        y_ret_val = y_ret[val_mask]
        
        y_vol_train = y_vol[train_mask]
        y_vol_val = y_vol[val_mask]
        
        print(f"    Train samples: {len(X_train)}")
        print(f"    Val samples: {len(X_val)}")
        print(f"    Train period: {X_train.index.min().date()} to {X_train.index.max().date()}")
        print(f"    Val period: {X_val.index.min().date()} to {X_val.index.max().date()}")
        
        # Verify no overlap
        if X_train.index.max() >= X_val.index.min():
            raise ValueError(
                f"❌ OVERLAP DETECTED! Train max ({X_train.index.max().date()}) >= "
                f"Val min ({X_val.index.min().date()})"
            )
        
        print(f"    ✅ No overlap: {(X_val.index.min() - X_train.index.max()).days} day gap")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train return model
        self.ret_model = xgb.XGBRegressor(**self.model_params)
        self.ret_model.fit(
            X_train_scaled, 
            y_ret_train,
            eval_set=[(X_val_scaled, y_ret_val)],            
            verbose=False
        )
        
        # Train volatility model
        self.vol_model = xgb.XGBRegressor(**self.model_params)
        self.vol_model.fit(
            X_train_scaled, 
            y_vol_train,
            eval_set=[(X_val_scaled, y_vol_val)],
            verbose=False
        )
        
        # Evaluate on validation set
        ret_pred = self.ret_model.predict(X_val_scaled)
        vol_pred = self.vol_model.predict(X_val_scaled)
        
        ret_r2 = r2_score(y_ret_val, ret_pred)
        vol_r2 = r2_score(y_vol_val, vol_pred)
        ret_mse = mean_squared_error(y_ret_val, ret_pred)
        vol_mse = mean_squared_error(y_vol_val, vol_pred)
        
        print(f"    Return Model  - R²: {ret_r2:.3f}, MSE: {ret_mse:.6f}")
        print(f"    Vol Model     - R²: {vol_r2:.3f}, MSE: {vol_mse:.6f}")
        
        # Store validation metrics for later analysis
        self.last_val_metrics = {
            'ret_r2': ret_r2,
            'vol_r2': vol_r2,
            'ret_mse': ret_mse,
            'vol_mse': vol_mse
        }

        self.is_trained = True
        return self

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        Xs = self.scaler.transform(X)
        vol = self.vol_model.predict(Xs)
        ret = self.ret_model.predict(Xs)

        vol = np.clip(vol, 1e-6, 2.0)
        ret = np.clip(ret, -1.0, 1.0)

        return {"vol": vol, "ret": ret}

    def predict_latest(self, X, symbols):
        """
        Only predict for the most recent row per symbol.
        Assumes X is time-ordered and contains multiple rows per symbol.
        """
        latest_rows = X.groupby(symbols).tail(1)
        preds = self.predict(latest_rows)

        return preds, latest_rows.index

    def optimal_weights(self, preds, assets, method="vol_parity"):
        vol = preds["vol"]
        ret = preds["ret"]

        if method == "vol_parity":
            score = 1 / (vol + 1e-6)
        else: # default to sharpe ratio
            score = ret / (vol + 1e-6)

        score = np.clip(score, -5, 5)

        w = np.maximum(score, 0)
        w = w / (w.sum() + 1e-9)

        return dict(zip(assets, w))

    @staticmethod
    def evaluate(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"r2": r2, "mse": mse}

    def save(self, path="portfolio_optimizer.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="portfolio_optimizer.pkl"):
        return joblib.load(path)