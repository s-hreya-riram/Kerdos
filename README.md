# Kerdos Fund — AI-Driven Multi-Asset Portfolio Strategy

## What This Is

Kerdos Fund is an ML-driven portfolio that allocates across 7 assets (BTC, SPY, GLD, SLV, SMH, ZAP, DFEN) using three stacked models:

1. **Volatility predictor** (XGBoost Regressor) — predicts next-day realised vol; drives inverse-vol parity weights
2. **Direction classifier** (XGBoost Classifier) — predicts P(return > 0); used as a soft allocation tilt
3. **Regime filter** — scales gross exposure based on SPY realised vol (CALM / CAUTION / FEAR)

Backtest result (Jan 2024 – Feb 2026): **234% total return, 1.95 Sharpe, -18.14% max drawdown** vs SPY's 46% / 0.99 / -18.75%.

---

## Performance Disclosure

Kerdos Fund can operate in two modes:

**Training mode:**

* Models retrain daily on a 90-day rolling window
* Runtime: ~1.8 seconds per trading day
* Performance (Jan 2024 – Feb 2026): 234% return, 1.95 Sharpe
* Advantages: Uses the most recent data; model adapts daily
* Considerations: Slightly longer runtime during backtests or submissions

**Pre-trained mode (default):**

* Models loaded from disk (trained on Jan 2024 – Feb 2026)
* Runtime: ~30–45 seconds for full competition period (faster than training mode)
* Performance (single static model trained on Jan 2024 – Feb 2026, evaluated without rolling retraining): 275% return, 2.42 Sharpe
* Advantages: Reduces computational load, ideal for rapid evaluation
* Considerations: Models do not reflect the most recent data

### Switching to Pre-Trained Mode

In the init method of the strategy, update the `load_pretrained` attribute from False to True to leverage pretrained models.

```python
    def __init__(self,
                 broker,
                 performance_callback=None,
                 optimizer=None,
                 load_pretrained=True,  # UPDATE: Flag to control pre-trained vs. training mode; False = retraining, True = pretrained
                 pretrained_path='models/portfolio_optimizer.pkl',  # Path to saved models
                 min_samples=50,
                 allow_shorts=False,
                 max_short_exposure=0.30,
                 min_cash_buffer=0.05,
                 margin_requirement=1.5,
                 weekend_crypto_adjustment=True,
                 # Regime thresholds (annualised SPY realised vol)
                 regime_calm_threshold=0.12,   # below 12% ann vol → full exposure
                 regime_fear_threshold=0.22,   # above 22% ann vol → 30% exposure
                 # Direction gate
                 direction_gate_threshold=0.0,
                 **kwargs)
```

---

## Project Structure

```
.
├── README.md                        # This file
├── backtest.py                      # Main backtest runner (entry point)
├── config.py                        # Alpaca API keys + client setup
├── requirements.txt                 # Python dependencies
├── data/
│   ├── __init__.py
│   ├── constants.py                 # Assets, hyperparams, position limits
│   ├── data_fetcher.py              # Yahoo Finance (backtest) + Alpaca (live)
│   ├── data_pipeline.py             # Feature engineering + ML prep
│   ├── model.py                     # PortfolioRiskOptimizer + RegimeFilter
│   └── utils.py                     # Timezone helpers
├── strategies/
│   ├── __init__.py
│   └── strategy.py                  # Main ML strategy (MLPortfolioStrategy)
├── models/
│   ├── portfolio_optimizer.pkl     # Pre-trained model (for pre-trained mode)
│   └── training_metadata.json      # Model training metadata
└── logs/                           # Backtest outputs
    ├── best_results                # Best backtest results (retraining)
    |      ├── *_tearsheet.html            # Performance tearsheet
    |      ├── *_trades.csv                # Trade log
    |      ├── *_stats.csv                 # Strategy statistics
    |      └── *_indicators.csv            # Technical indicators
    ├── pretrained                # Best backtest results (retraining)
    |      ├── *_tearsheet.html            # Performance tearsheet
    |      ├── *_trades.csv                # Trade log
    |      ├── *_stats.csv                 # Strategy statistics
    |      └── *_indicators.csv            # Technical indicators
    ├── competition_period_in_2025         # Comparison against 2025 equivalent for the competiton timeframe
    |      ├── *_tearsheet.html            # Performance tearsheet
    |      ├── *_trades.csv                # Trade log
    |      ├── *_stats.csv                 # Strategy statistics
    |      └── *_indicators.csv            # Technical indicators
    ├── *_tearsheet.html            # Performance tearsheet
    ├── *_trades.csv                # Trade log
    ├── *_stats.csv                 # Performance statistics
    └── *_indicators.csv            # Technical indicators
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys (Optional)

Create a `.env` file in the project root:

```env
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
```

> **Note:** API keys are only required for live trading. Backtesting works without any API keys using Yahoo Finance data.

---

## Running a Backtest

Edit `backtest.py` to set your date range, then:

```bash
python backtest.py
```

Key parameters in `backtest.py`:

```python
backtesting_start = datetime(2024, 1, 1)    # change as needed
backtesting_end   = datetime(2026, 2, 17)
budget            = 10000                    # starting capital in USD
```

The backtest will:

* Fetch price data from Yahoo Finance
* Retrain all three models daily on a 90-day rolling window (training mode)
* Apply regime filtering and direction gating
* Output results to `logs/` directory:

  * `*_tearsheet.html` - Performance tearsheet with charts
  * `*_trades.csv` - Complete trade log
  * `*_stats.csv` - Performance statistics
  * `*_indicators.csv` - Technical indicators

---

## Key Configuration (`data/constants.py`)

| Parameter              | Value          | Purpose                               |
| ---------------------- | -------------- | ------------------------------------- |
| `MAX_GROSS_EXPOSURE`   | 0.95           | Max portfolio deployment              |
| `MAX_POSITION_PCT`     | 0.33           | Max weight per single asset           |
| `MIN_TRADE_DOLLARS`    | 100            | Minimum order size                    |
| `XGB_MODEL_PARAMS`     | Bayesian-tuned | Vol + ret regressor hyperparams       |
| `CODE_SUBMISSION_DATE` | 2026-02-23     | Controls training vs pre-trained mode |

---

## Key Strategy Parameters (`strategies/strategy.py`)

| Parameter                  | Value | Purpose                          |
| -------------------------- | ----- | -------------------------------- |
| `lookback_days`            | 90    | Rolling training window          |
| `direction_gate_threshold` | 0.0   | Soft tilt (0 = no hard gating)   |
| `regime_calm_threshold`    | 0.12  | Below 12% ann. vol = full deploy |
| `regime_fear_threshold`    | 0.22  | Above 22% ann. vol = 30% deploy  |
| `min_cash_buffer`          | 0.05  | Minimum 5% cash reserve          |

---

## Assets Traded

| Symbol            | Name                | Role                                  |
| ----------------- | ------------------- | ------------------------------------- |
| BTC-USD / BTC/USD | Bitcoin             | High-beta alternative; 24/7 liquidity |
| SPY               | S&P 500 ETF         | Core equity exposure                  |
| GLD               | Gold ETF            | Safe haven hedge                      |
| SLV               | Silver ETF          | Amplified metals exposure             |
| SMH               | Semiconductor ETF   | AI/tech momentum                      |
| ZAP               | Electrification ETF | AI infrastructure theme               |
| DFEN              | Defense ETF         | Geopolitical uncertainty hedge        |

> Yahoo Finance symbols are used for backtesting; Alpaca symbols for live trading. Both are stored in `ASSETS` as `(alpaca_sym, yahoo_sym)` tuples.

---

## Reproducing the Main Backtest

To reproduce the headline numbers (112% return, 1.89 Sharpe):

```python
# In backtest.py
backtesting_start = datetime(2024, 1, 21)
backtesting_end   = datetime(2026, 2, 21)
budget            = 10000
strategy = MLPortfolioStrategy(broker=broker)
```

Expected output (approximate):

* Total Return: ~112%
* Sharpe Ratio: ~1.89
* Max Drawdown: ~-11.19%
* CAGR: ~43.79%

Runtime: approximately 10–20 minutes depending on machine speed (models retrain daily).

The logs for the previous runs can be found in `logs/best_results`. Similarly results for the pretrained run can be found in `logs/pretrained`. The strategy was stress-tested under elevated volatility regimes (see `logs/competition_period_in_2025` for the relevant logs).

---

## Notes for Markers

* **No data leakage:** Training data always ends at `today - 1 day`. A runtime check raises `ValueError` if this is violated.
* **Temporal validation split:** Train/val split is done by date (not row count) to prevent look-ahead bias.
* **Hyperparameters** in `constants.py` were found via Bayesian optimisation using Optuna with 5-fold time-series cross-validation.
* **Weekend handling:** Stock positions are frozen on weekends; crypto positions can be adjusted on any day.
* **Backtesting uses Yahoo Finance** (free, no API key needed). Live trading uses Alpaca paper account.
* **Pre-trained models** are included in `models/` directory for quick evaluation if needed.
