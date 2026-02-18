# Kerdos Fund — AI-Driven Multi-Asset Portfolio Strategy

## What This Is

Kerdos Fund is an ML-driven portfolio that allocates across 7 assets (BTC, SPY, GLD, SLV, SMH, ZAP, DFEN) using three stacked models:

1. **Volatility predictor** (XGBoost Regressor) — predicts next-day realised vol; drives inverse-vol parity weights
2. **Direction classifier** (XGBoost Classifier) — predicts P(return > 0); used as a soft allocation tilt
3. **Regime filter** — scales gross exposure based on SPY realised vol (CALM / CAUTION / FEAR)

Backtest result (Jan 2024 – Feb 2026): **117% total return, 1.87 Sharpe, -11.8% max drawdown** vs SPY's 48% / 1.01 / -18.75%.

---

## Project Structure

```
.
├── config.py                        # Alpaca API keys + client setup
├── run_backtest.py                  # Run a backtest (entry point)
├── streamlit_app.py                 # Live dashboard
├── data/
│   ├── constants.py                 # Assets, hyperparams, position limits
│   ├── data_fetcher.py              # Yahoo Finance (backtest) + Alpaca (live)
│   ├── data_pipeline.py             # Feature engineering + ML prep
│   ├── model.py                     # PortfolioRiskOptimizer + RegimeFilter
│   └── utils.py                     # Timezone helpers
├── strategies/
│   ├── xgboost_strategy.py          # Main ML strategy (MLPortfolioStrategy)
│   └── strategy.py                  # SPY buy-and-hold benchmark
├── hyperparameter_tuning/
│   └── hyperparameter_tuning.py     # Bayesian optimisation (Optuna)
└── assets/
    └── logo.png                     # Kerdos logo (used in dashboard)
```

---

## Setup

### 1. Install dependencies defined in requirements.txt

```
pip install requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret

SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=PRODUCTION
SNOWFLAKE_ROLE=ACCOUNTADMIN
```

> **Note:** Alpaca keys must be for a **paper trading** account. The strategy uses `PAPER: True` in `config.py`.  
> Snowflake is only required for the live dashboard. Backtesting works without it.

---

## Running a Backtest

Edit `run_backtest.py` to set your date range, then:

```bash
python run_backtest.py
```

Key parameters in `run_backtest.py`:

```python
backtesting_start = datetime(2024, 1, 1)    # change as needed
backtesting_end   = datetime(2026, 2, 17)
budget            = 10000                    # starting capital in USD
```

The backtest will:
- Fetch price data from Yahoo Finance
- Retrain all three models daily on a 90-day rolling window
- Apply regime filtering and direction gating
- Output a tearsheet HTML + trades CSV to the working directory

---

## Running the Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard requires Snowflake credentials in `.env`. It shows:
- Live portfolio allocation (pie chart, fetched from Snowflake)
- Performance vs SPY benchmark
- Drawdown analysis
- Key metrics table

---

## Key Configuration (`data/constants.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MAX_GROSS_EXPOSURE` | 0.95 | Max portfolio deployment |
| `MAX_POSITION_PCT` | 0.33 | Max weight per single asset |
| `MIN_TRADE_DOLLARS` | 100 | Minimum order size |
| `XGB_MODEL_PARAMS` | Bayesian-tuned | Vol + ret regressor hyperparams |

---

## Key Strategy Parameters (`xgboost_strategy.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `lookback_days` | 90 | Rolling training window |
| `direction_gate_threshold` | 0.0 | Soft tilt (0 = no hard gating) |
| `regime_calm_threshold` | 0.12 | Below 12% ann. vol = full deploy |
| `regime_fear_threshold` | 0.22 | Above 22% ann. vol = 30% deploy |
| `min_cash_buffer` | 0.05 | Minimum 5% cash reserve |

---

## Assets Traded

| Symbol | Name | Role |
|--------|------|------|
| BTC-USD / BTC/USD | Bitcoin | High-beta alternative; 24/7 liquidity |
| SPY | S&P 500 ETF | Core equity exposure |
| GLD | Gold ETF | Safe haven hedge |
| SLV | Silver ETF | Amplified metals exposure |
| SMH | Semiconductor ETF | AI/tech momentum |
| ZAP | Electrification ETF | AI infrastructure theme |
| DFEN | Defense ETF | Geopolitical uncertainty hedge |

> Yahoo Finance symbols are used for backtesting; Alpaca symbols for live trading. Both are stored in `ASSETS` as `(alpaca_sym, yahoo_sym)` tuples.

---

## Reproducing the Main Backtest

To reproduce the headline numbers (117% return, 1.87 Sharpe):

```python
# In run_backtest.py
backtesting_start = datetime(2024, 1, 1)
backtesting_end   = datetime(2026, 2, 17)
budget            = 10000
strategy = MLPortfolioStrategy(broker=broker, allow_shorts=True)
```

Expected output (approximate):
- Total Return: ~117%
- Sharpe Ratio: ~1.87
- Max Drawdown: ~-11.8%
- CAGR: ~44.1%

Runtime: approximately 10–20 minutes depending on machine speed (models retrain daily).

---

## Notes for Markers

- **No data leakage:** Training data always ends at `today - 1 day`. A runtime check raises `ValueError` if this is violated.
- **Temporal validation split:** Train/val split is done by date (not row count) to prevent look-ahead bias.
- **Hyperparameters** in `constants.py` were found via Bayesian optimisation (`hyperparameter_tuning/hyperparameter_tuning.py`) using Optuna with 5-fold time-series cross-validation.
- **Weekend handling:** Stock positions are frozen on weekends; crypto positions can be adjusted 24/7.
- **Backtesting uses Yahoo Finance** (free, no API key needed). Live trading uses Alpaca paper account.