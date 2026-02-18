import xgboost as xgb
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score
from data.constants import XGB_MODEL_PARAMS, DIRECTION_GATE_MODEL_PARAMS

class RegimeFilter:
    """
    Market-wide fear/greed signal derived from SPY realised volatility
    (a VIX proxy usable without a separate data feed).

    Logic:
        - Compute rolling 21-day annualised vol of SPY daily returns
        - Classify into three regimes:
            CALM   (vol < calm_threshold)   → full allocation allowed
            CAUTION(calm <= vol < fear_thr) → allocation scaled down
            FEAR   (vol >= fear_threshold)  → allocation capped heavily
    """

    def __init__(self, calm_threshold=0.12, fear_threshold=0.22):
        """
        Args:
            calm_threshold:  annualised vol below which market is "calm"  (~12%)
            fear_threshold:  annualised vol above which market is "fearful" (~22%)
        """
        self.calm_threshold = calm_threshold
        self.fear_threshold = fear_threshold

    def compute(self, spy_prices: pd.Series, window: int = 21) -> dict:
        """
        Args:
            spy_prices: Series of SPY close prices (most recent last)
            window:     look-back for realised vol (trading days)

        Returns:
            dict with keys:
                regime           str    "CALM" | "CAUTION" | "FEAR"
                realised_vol     float  current annualised vol
                allocation_scale float  multiplier to apply to gross exposure (0–1)
                vix_proxy        float  same as realised_vol, named for clarity
        """
        returns = spy_prices.pct_change().dropna()
        if len(returns) < window:
            return {"regime": "CALM", "realised_vol": 0.15,
                    "allocation_scale": 1.0, "vix_proxy": 0.15}

        realised_vol = returns.iloc[-window:].std() * np.sqrt(252)

        if realised_vol < self.calm_threshold:
            regime = "CALM"
            scale = 1.0
        elif realised_vol < self.fear_threshold:
            # Linear interpolation between 1.0 and 0.5 across the caution band
            band = self.fear_threshold - self.calm_threshold
            position = (realised_vol - self.calm_threshold) / band
            scale = 1.0 - 0.5 * position
            regime = "CAUTION"
        else:
            # Hard cap: only use 30% of gross exposure in fear regime
            scale = 0.30
            regime = "FEAR"

        return {
            "regime": regime,
            "realised_vol": float(realised_vol),
            "allocation_scale": float(scale),
            "vix_proxy": float(realised_vol),
        }


class PortfolioRiskOptimizer:
    """
    ML-based portfolio optimizer for time series.

    Trains three models on every call to fit():
        1. vol_model  (XGBRegressor)   — predicts next-day realised volatility
        2. ret_model  (XGBRegressor)   — predicts next-day return magnitude
           (kept for diagnostics; not used in weight computation by default)
        3. dir_model  (XGBClassifier)  — predicts direction: 1 = up, 0 = down

    Weight computation (vol_parity + direction filter):
        base_score   = 1 / predicted_vol          (inverse vol parity)
        gated_score  = base_score * direction_prob (suppress bearish assets)
        weight       = gated_score / sum(gated_scores)  → sums to 1.0

    optimal_weights() returns weights normalised to sum=1.0 (relative allocations).
    The caller (strategy) applies MAX_GROSS_EXPOSURE and regime_scale exactly once.
    This prevents the double-application bug where scale² crushes deployment.
    """

    def __init__(self, risk_target=0.15, model_params=None, classifier_params=None,
                 direction_gate_threshold=0.50):
        self.risk_target = risk_target
        self.scaler = StandardScaler()
        self.vol_model = None
        self.ret_model = None
        self.dir_model = None
        self.is_trained = False

        self.model_params = model_params or XGB_MODEL_PARAMS
        self.classifier_params = classifier_params or DIRECTION_GATE_MODEL_PARAMS
        self.direction_gate_threshold = direction_gate_threshold

        self.last_val_metrics = None
        self.last_regime_info = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y_ret, y_vol, validation_split=0.2):
        """
        Train all three models with a temporal train/validation split.

        The direction target y_dir is derived from y_ret internally:
            y_dir = 1  if y_ret > 0  else  0

        Args:
            X:                DataFrame with DatetimeIndex, one row per (date, symbol)
            y_ret:            Series, next-day return (regression target)
            y_vol:            Series, next-day volatility (regression target)
            validation_split: fraction of most-recent TIME PERIOD held out
        """
        if len(X) < 100:
            print(f"  ⚠️  Only {len(X)} samples — skipping validation split")
            Xs = self.scaler.fit_transform(X)
            y_dir = (y_ret > 0).astype(int)

            self.vol_model = xgb.XGBRegressor(**self.model_params)
            self.ret_model = xgb.XGBRegressor(**self.model_params)
            self.dir_model = xgb.XGBClassifier(**self.classifier_params)

            self.vol_model.fit(Xs, y_vol)
            self.ret_model.fit(Xs, y_ret)
            self.dir_model.fit(Xs, y_dir)

            self.is_trained = True
            return self

        # ---- Temporal split (no row-count split — avoids leakage) ----
        unique_dates = X.index.unique().sort_values()
        split_idx = int(len(unique_dates) * (1 - validation_split))
        train_end_date = unique_dates[split_idx - 1]
        val_start_date = unique_dates[split_idx]

        print(f"    🕐 Total unique dates : {len(unique_dates)}")
        print(f"    📅 Train cutoff       : {train_end_date.date()}")
        print(f"    📅 Val start          : {val_start_date.date()}")

        train_mask = X.index <= train_end_date
        val_mask   = X.index >= val_start_date

        X_train, X_val         = X[train_mask],     X[val_mask]
        y_ret_train, y_ret_val = y_ret[train_mask], y_ret[val_mask]
        y_vol_train, y_vol_val = y_vol[train_mask], y_vol[val_mask]

        # Derive direction targets
        y_dir_train = (y_ret_train > 0).astype(int)
        y_dir_val   = (y_ret_val   > 0).astype(int)

        print(f"    Train : {len(X_train)} rows  "
              f"({X_train.index.min().date()} → {X_train.index.max().date()})")
        print(f"    Val   : {len(X_val)} rows  "
              f"({X_val.index.min().date()} → {X_val.index.max().date()})")

        if X_train.index.max() >= X_val.index.min():
            raise ValueError(
                f"❌ OVERLAP! Train max ({X_train.index.max().date()}) >= "
                f"Val min ({X_val.index.min().date()})"
            )
        print(f"    ✅ Gap : {(X_val.index.min() - X_train.index.max()).days}d")

        # ---- Class balance info (useful for diagnosis) ----
        up_frac = y_dir_train.mean()
        print(f"    📊 Direction balance  : {up_frac:.1%} up / {1-up_frac:.1%} down (train)")

        # ---- Scale ----
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)

        # ---- Vol model ----
        self.vol_model = xgb.XGBRegressor(**self.model_params)
        self.vol_model.fit(
            X_train_s, y_vol_train,
            eval_set=[(X_val_s, y_vol_val)],
            verbose=False
        )

        # ---- Ret model ----
        self.ret_model = xgb.XGBRegressor(**self.model_params)
        self.ret_model.fit(
            X_train_s, y_ret_train,
            eval_set=[(X_val_s, y_ret_val)],
            verbose=False
        )

        # ---- Direction classifier ----
        self.dir_model = xgb.XGBClassifier(**self.classifier_params)
        self.dir_model.fit(
            X_train_s, y_dir_train,
            eval_set=[(X_val_s, y_dir_val)],
            verbose=False
        )

        # ---- Validation metrics ----
        ret_pred = self.ret_model.predict(X_val_s)
        vol_pred = self.vol_model.predict(X_val_s)
        dir_prob = self.dir_model.predict_proba(X_val_s)[:, 1]
        dir_pred = (dir_prob >= 0.5).astype(int)

        ret_r2  = r2_score(y_ret_val, ret_pred)
        vol_r2  = r2_score(y_vol_val, vol_pred)
        dir_acc = accuracy_score(y_dir_val, dir_pred)

        try:
            dir_auc = roc_auc_score(y_dir_val, dir_prob)
        except ValueError:
            dir_auc = float("nan")

        print(f"    Vol model  — R²  : {vol_r2:.3f}   MSE: {mean_squared_error(y_vol_val, vol_pred):.6f}")
        print(f"    Ret model  — R²  : {ret_r2:.3f}   MSE: {mean_squared_error(y_ret_val, ret_pred):.6f}")
        print(f"    Dir model  — Acc : {dir_acc:.3f}   AUC: {dir_auc:.3f}")

        self.last_val_metrics = {
            "ret_r2": ret_r2,
            "vol_r2": vol_r2,
            "dir_acc": dir_acc,
            "dir_auc": dir_auc,
            "ret_mse": float(mean_squared_error(y_ret_val, ret_pred)),
            "vol_mse": float(mean_squared_error(y_vol_val, vol_pred)),
        }

        self.is_trained = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X) -> dict:
        """
        Returns dict with keys:
            vol   np.ndarray  predicted next-day volatility
            ret   np.ndarray  predicted next-day return
            dir   np.ndarray  P(return > 0) from classifier
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained — call fit() first")

        Xs = self.scaler.transform(X)

        vol = np.clip(self.vol_model.predict(Xs), 1e-6, 2.0)
        ret = np.clip(self.ret_model.predict(Xs), -1.0, 1.0)
        dir_prob = self.dir_model.predict_proba(Xs)[:, 1]  # P(up)

        return {"vol": vol, "ret": ret, "dir": dir_prob}

    def predict_latest(self, X, symbols) -> tuple[dict, np.ndarray]:
        """
        Predict for the most recent row per symbol.

        Args:
            X:       Feature DataFrame (DatetimeIndex, multiple rows per symbol)
            symbols: Series of symbol strings, aligned with X's rows

        Returns:
            preds            dict        {vol, ret, dir} arrays, one entry per symbol
            returned_symbols np.ndarray  symbol name per prediction, same order

        Why not return latest_rows.index?
            The DatetimeIndex is shared across symbols (many symbols per date),
            so using it to re-index symbols_clean with .loc produces duplicates
            and causes "index N out of bounds for size N" in the caller.
            Returning symbol names directly is unambiguous.
        """
        # Attach symbol labels so they survive the groupby.
        # reset_index(drop=True) on both ensures positional alignment
        # even when X and symbols come from a masked subset with gaps in the index.
        X_with_sym = X.reset_index(drop=True).copy()
        sym_array  = np.asarray(symbols) if not isinstance(symbols, np.ndarray) else symbols
        X_with_sym["_symbol"] = sym_array

        # tail(1) per symbol — result is in alphabetical group order
        latest_rows      = X_with_sym.groupby("_symbol").tail(1)
        returned_symbols = latest_rows["_symbol"].values
        feature_cols     = latest_rows.drop(columns=["_symbol"])

        preds = self.predict(feature_cols)
        return preds, returned_symbols

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def optimal_weights(self, preds: dict, assets: np.ndarray,
                        method: str = "vol_parity") -> dict:
        """
        Compute normalised target portfolio weights.

        Returns weights that sum to 1.0 (relative allocations only).
        The caller (strategy) is responsible for applying:
            - MAX_GROSS_EXPOSURE  (overall deployment cap)
            - regime_scale        (fear/calm scaling from RegimeFilter)

        Keeping both multiplications in the strategy — and out of here —
        prevents the double-application bug (scale²) that kept the
        portfolio at $10,000 during non-CALM regimes.

        Args:
            preds:   dict from predict() — must contain 'vol', 'ret', 'dir'
            assets:  array of symbol names aligned with preds arrays
            method:  'vol_parity'  (recommended)
                     'sharpe'      (noisier, ret/vol ratio)

        Returns:
            dict  {symbol: weight}  normalised to sum to 1.0
        """
        vol = preds["vol"]
        ret = preds["ret"]
        dir_prob = preds.get("dir", np.ones(len(vol)) * 0.5)

        # ---- Base score ----
        if method == "vol_parity":
            base_score = 1.0 / (vol + 1e-6)
        else:  # sharpe
            base_score = ret / (vol + 1e-6)

        base_score = np.clip(base_score, 0, 5)

        # ---- Direction gate (soft) ----
        # Hard threshold zeroes out assets below direction_gate_threshold.
        # Above the threshold, weights are amplified proportional to P(up),
        # so a confident bullish asset wins more allocation than a marginal one.
        hard_mask = np.where(dir_prob >= self.direction_gate_threshold, 1.0, 0.0)
        gated_score = base_score * dir_prob * hard_mask

        # ---- Normalise to sum=1 ----
        total = gated_score.sum()
        if total < 1e-9:
            # All assets gated — fall back gracefully rather than going to cash.
            # Equal weight among assets with P(up) >= 0.40, or truly equal if none.
            fallback_mask = dir_prob >= 0.40
            gated_score = (fallback_mask.astype(float) if fallback_mask.any()
                           else np.ones(len(vol)))
            total = gated_score.sum()

        weights = gated_score / total  # sums to 1.0; gross exposure applied by caller

        return dict(zip(assets, weights.tolist()))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "r2":  r2_score(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
        }

    def save(self, path="portfolio_optimizer.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="portfolio_optimizer.pkl"):
        return joblib.load(path)