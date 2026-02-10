name: Daily Backtest Update

on:
  # Run daily at 6 PM EST (after market close at 4 PM + buffer)
  schedule:
    - cron: '0 23 * * 1-5'  # 11 PM UTC = 6 PM EST (weekdays only)
  
  # Allow manual triggering
  workflow_dispatch:

jobs:
  run-incremental-backtest:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Test Snowflake connection
        env:
          SNOWFLAKE_USER: ${{ secrets.SNOWFLAKE_USER }}
          SNOWFLAKE_PASSWORD: ${{ secrets.SNOWFLAKE_PASSWORD }}
          SNOWFLAKE_ACCOUNT: ${{ secrets.SNOWFLAKE_ACCOUNT }}
          SNOWFLAKE_WAREHOUSE: ${{ secrets.SNOWFLAKE_WAREHOUSE }}
          SNOWFLAKE_DATABASE: ${{ secrets.SNOWFLAKE_DATABASE }}
          SNOWFLAKE_SCHEMA: ${{ secrets.SNOWFLAKE_SCHEMA }}
        run: |
          cd streamlit
          python test_snowflake_connection.py
      
      - name: Run incremental backtest
        env:
          # Snowflake credentials from GitHub Secrets
          SNOWFLAKE_USER: ${{ secrets.SNOWFLAKE_USER }}
          SNOWFLAKE_PASSWORD: ${{ secrets.SNOWFLAKE_PASSWORD }}
          SNOWFLAKE_ACCOUNT: ${{ secrets.SNOWFLAKE_ACCOUNT }}
          SNOWFLAKE_WAREHOUSE: ${{ secrets.SNOWFLAKE_WAREHOUSE }}
          SNOWFLAKE_DATABASE: ${{ secrets.SNOWFLAKE_DATABASE }}
          SNOWFLAKE_SCHEMA: ${{ secrets.SNOWFLAKE_SCHEMA }}
          
          # Alpaca credentials (for broker initialization)
          ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
          ALPACA_API_SECRET: ${{ secrets.ALPACA_API_SECRET }}
          ALPACA_PAPER: ${{ secrets.ALPACA_PAPER }}
        run: |
          cd streamlit
          python run_incremental_backtest.py
        continue-on-error: false
      
      - name: Upload logs on failure
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: backtest-logs
          path: |
            streamlit/*.log
            streamlit/backtest_output/
          retention-days: 7
      
      - name: Notify on success
        if: success()
        run: |
          echo "✅ Incremental backtest completed successfully"
          echo "📊 Snowflake data updated with latest market data"
      
      - name: Notify on failure
        if: failure()
        run: |
          echo "❌ Incremental backtest failed"
          echo "Check the logs in artifacts for details"
          echo "Common issues:"
          echo "  1. Snowflake warehouse not activated"
          echo "  2. Network timeout"
          echo "  3. Missing market data for weekend/holiday"