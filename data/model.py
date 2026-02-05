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
    Predicts future volatility and returns, converts predictions into portfolio weights.
    Includes time-series-aware train/validation/holdout splits, CV, and evaluation.
    """

    def __init__(self, risk_target=0.15, model_params=None):
        self.risk_target = risk_target
        self.scaler = StandardScaler()
        self.vol_model = None
        self.ret_model = None
        self.is_trained = False

        # TODO perform hyperparameter tuning to identify the best inputs
        # for the model_params
        self.model_params = model_params or XGB_MODEL_PARAMS

        self.last_val_metrics = None

    
    def fit(self, X, y_ret, y_vol, validation_split=0.2):
        """
        Train with time-series validation split
        
        Args:
            X: Features (DataFrame)
            y_ret: Return targets (Series)
            y_vol: Volatility targets (Series)
            validation_split: Fraction of most recent data for validation
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
        
        # Splitting data w/o shuffling to retain time series order
        # Last fraction of data (corresponding to validation_split) is for validation
        split_idx = int(len(X) * (1 - validation_split))
        
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        
        y_ret_train = y_ret.iloc[:split_idx]
        y_ret_val = y_ret.iloc[split_idx:]
        
        y_vol_train = y_vol.iloc[:split_idx]
        y_vol_val = y_vol.iloc[split_idx:]
        
        print(f"    Train samples: {len(X_train)}")
        print(f"    Val samples: {len(X_val)}")
        print(f"    Train period: {X_train.index.min().date()} to {X_train.index.max().date()}")
        print(f"    Val period: {X_val.index.min().date()} to {X_val.index.max().date()}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train return model with early stopping
        self.ret_model = xgb.XGBRegressor(**self.model_params)
        self.ret_model.fit(
            X_train_scaled, 
            y_ret_train,
            eval_set=[(X_val_scaled, y_ret_val)],            
            verbose=False
        )
        
        # Train volatility model with early stopping
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

    def optimal_weights(self, preds, assets, method="sharpe"):
        vol = preds["vol"]
        ret = preds["ret"]

        if method == "vol_parity":
            score = 1 / (vol + 1e-6)
        else:
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

# Testing with simulated data to check functionality before wiring up in the strategy
if __name__ == "__main__":
    print("PortfolioRiskOptimizer Example")
    # Example data
    n_samples = 1000
    n_features = 10
    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f"feat_{i}" for i in range(n_features)])
    y_ret = pd.Series(np.random.randn(n_samples) * 0.01)
    y_vol = pd.Series(np.random.rand(n_samples) * 0.05)
    assets = [f"asset_{i}" for i in range(n_samples)]
    # Initialize and train model
    optimizer = PortfolioRiskOptimizer()
    optimizer.fit(X, y_ret, y_vol)
    # Predict
    preds = optimizer.predict(X)
    # Get optimal weights
    weights = optimizer.optimal_weights(preds, assets)
    print("Optimal Weights:", weights)
    # Evaluate
    ret_eval = optimizer.evaluate(y_ret, preds["ret"])
    vol_eval = optimizer.evaluate(y_vol, preds["vol"])
    print("Return Evaluation:", ret_eval)
    print("Volatility Evaluation:", vol_eval)