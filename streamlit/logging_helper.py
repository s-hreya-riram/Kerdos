import pandas as pd
from datetime import datetime


def build_daily_performance(strategy, strategy_name):

    return {
        "TIMESTAMP": datetime.utcnow(),
        "STRATEGY_NAME": strategy_name,
        "PORTFOLIO_VALUE": strategy.get_portfolio_value(),
        "CASH": strategy.get_cash(),
    }


def build_daily_positions(strategy, strategy_name):

    portfolio_value = strategy.get_portfolio_value()
    timestamp = datetime.utcnow()

    rows = []

    for position in strategy.get_positions():

        symbol = position.symbol
        value = position.quantity * position.last_price

        weight = value / portfolio_value if portfolio_value > 0 else 0

        rows.append({
            "TIMESTAMP": timestamp,
            "STRATEGY_NAME": strategy_name,
            "SYMBOL": symbol,
            "POSITION_VALUE": value,
            "WEIGHT": weight
        })

    return rows
