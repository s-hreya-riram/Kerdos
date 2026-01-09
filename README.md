## QF5208-AI-and-FinTech-AY2526SEM2 – Trading Strategy Assignment

### 1. Assignment overview

In this assignment you will design, implement, and evaluate a systematic trading strategy using LumiBot and Alpaca.

Your core tasks are to:

- **Design and implement your own trading / portfolio strategy** in `strategies/strategy.py`.
- **Backtest your strategy** over a historical period.
- (Optional) **run it in paper-trading mode** on a live market feed.

You will use:

- **LumiBot** to manage strategies, brokers, and backtests.
- **Alpaca** as a broker interface (paper account).
- **Yahoo Finance** data (via LumiBot) for historical prices in backtests or possible model training.

---

### 2. Introduction

- **LumiBot**
  - An open-source **algorithmic trading framework** in Python.
  - Docs: [LumiBot documentation](https://lumibot.lumiwealth.com/index.html).

- **Alpaca**
  - A broker with a well-documented **trading API**.
  - Website: [https://alpaca.markets](https://alpaca.markets).

In this repository, LumiBot + Alpaca are already wired up for you; you focus on **strategy logic**.

---

### 3. Repository structure

- **`backtest.py`**
  - Runs a **backtest** of a strategy using historical data from Yahoo Finance.
  - You will adapt this to backtest **your** `Strategy` from `strategies/strategy.py`.

- **`strategies/strategy.py`**
  - The **main file you will edit**.
  - Contains a subclass of `lumibot.strategies.Strategy` named `Strategy`.
  - You will implement your logic primarily in:
    - `initialize(self)`
    - `on_trading_iteration(self)`

- **`strategies/example_strategy_1.py` – `example_strategy_5.py`**
  - Example strategies provided as **reference**:
    - Simple daily DCA into SPY.
    - Buy-and-hold SPY.
    - Equal-weight MAG7 portfolio.
    - Permanent portfolio / all‑weather allocation.
    - Simple ML-based BTC strategy using logistic regression.
  - You are encouraged to read these files to understand LumiBot’s API and coding style.

---

### 4. Setup instructions

#### 4.1. Prerequisites

- **Python 3.10+** (recommended)
- A **code editor** (VS Code, PyCharm, Cursor, etc.)
- Internet access

#### 4.2. Create an Alpaca paper trading account

1. Go to the Alpaca website ([https://alpaca.markets](https://alpaca.markets)) and create an account (if you already have one, we recommend registering a new one to separate it from any personal account).
2. In your account dashboard, enable a **paper trading** account.
3. Generate an API key and secret for paper trading.
4. Note down:
   - `ALPACA_API_KEY`
   - `ALPACA_API_SECRET`

#### 4.3. Clone this repository or download the code. Open the folder in your editor.

#### 4.4. (Recommended) Create and activate a virtual environment

Create a **virtual environment** so that assignment packages do not interfere with your global Python installation.

- **On Windows (PowerShell or Command Prompt)**:

```bash
python -m venv .venv
.venv\Scripts\activate
```

- **On macOS / Linux**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the beginning of your terminal prompt. Remember to activate it each time you start a new terminal session.

#### 4.5. Install Python dependencies

With the virtual environment **activated**, run:

```bash
python -m pip install -r requirements.txt
```

This will install all required libraries.

#### 4.6. Create and modify your `.env` file

Create a file named `.env` in the project root. Add the following lines, replacing the placeholders with your actual Alpaca API key and secret:

```bash
ALPACA_API_KEY=YOUR_ALPACA_API_KEY
ALPACA_API_SECRET=YOUR_ALPACA_API_SECRET
```

The file `config.py` will load these values and build `ALPACA_CONFIG` for both backtesting and paper trading.

---

### 5. Assignment tasks

#### 5.1. Implement your strategy in `strategies/strategy.py`

- Open `strategies/strategy.py`.
- You will see a class definition:

```python
from lumibot.strategies import Strategy


class Strategy(Strategy):
    ...
```

This is your strategy class (a subclass of LumiBot’s `Strategy` base class).

You should:

- Implement **`initialize(self)`**.
- Implement **`on_trading_iteration(self)`**.

You may add **helper methods** or additional modules to keep your code organized.

#### 5.2. Backtest your strategy

1. Open `backtest.py`.
2. Ensure it imports your `Strategy` if needed.
3. You can set the **backtest period** and **initial budget** (or other attributes) to experiment with your strategy under different market conditions.
4. Run the backtest from the project root:

```bash
python backtest.py  # make sure your .venv if you set up the virtual environment
```

The strategy will run in a simulated environment using historical data (from Yahoo Finance in this case).
5. Two HTML files will open, showing the analysis of the backtest. You can also track the trading actions in the `logs` folder.

#### 5.3. (Optional) Paper trade your strategy

Once your strategy is stable and you understand its behavior in backtests:

1. Open `paper_trade.py` and verify it uses your `Strategy` from `strategies/strategy.py`.
2. Run:

```bash
python paper_trade.py
```

This will:

- Connect to Alpaca’s **paper trading API** using the keys from `.env`.
- Execute your strategy logic in near real time, sending orders to the paper account.
- Make sure that you are using paper money instead of real money. **DO NOT DEPOSIT YOUR REAL MONEY INTO THIS ACCOUNT.**

---

### 6. Suggested workflow

- **Step 1**: Read the example strategies in `strategies/` to understand LumiBot’s API.
- **Step 2**: Start simple (think about moving average, momentum, mean reversion, factor-based, or a simple ML signal).
- **Step 3**: Implement your logic in `strategies/strategy.py`.
- **Step 4**: Backtest with `backtest.py` over one or more historical periods.
- **Step 5**: Analyze performance (returns, drawdowns, Sharpe, turnover, etc.) and refine.

---

### 7. What to submit

- **Code**:
  - Your completed `strategies/strategy.py`.
  - Any additional helper modules (if allowed).
- **Backtest results**:
  - The `logs` folder (or selected log files) from your best-performing backtest, clearly labeled.

---

Your goal is not just to “make money” in a backtest, but to **demonstrate understanding** of:

- Systematic strategy design,
- Proper use of trading APIs and backtesting tools,
- And critical evaluation of quantitative trading results.

Enjoy your exploration of trading.

> This repository is for course instruction only. It does not provide investment advice or any recommendation to trade.

