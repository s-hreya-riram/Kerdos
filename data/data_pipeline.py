import pandas as pd
import numpy as np
from .utils import to_utc
from config import STOCK_CLIENT, CRYPTO_CLIENT
from .constants import CRYPTO_SYMBOLS

# Import Alpaca modules
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

def fetch_historical_data(symbols, start_date, end_date=None, timeframe=TimeFrame.Day, feed=None):
    print(f"Fetching historical data for: {symbols}")

    start_time = to_utc(start_date)
    end_time = to_utc(end_date) if end_date else None

    # Separate stock and crypto symbols
    stock_symbols = [s for s in symbols if not CRYPTO_SYMBOLS(s)]
    crypto_symbols = [s for s in symbols if CRYPTO_SYMBOLS(s)]

    data_dict = {}

    # Retrieve historical stock data
    if stock_symbols:
        request = StockBarsRequest(
            symbol_or_symbols=stock_symbols,
            timeframe=timeframe,
            start=start_time,
            end=end_time,
            feed=feed
        )
        bars = STOCK_CLIENT.get_stock_bars(request)

        for symbol in stock_symbols:
            if symbol in bars.data and len(bars.data[symbol]) > 0:
                b = bars.data[symbol]
                df = pd.DataFrame({
                    "Open": [x.open for x in b],
                    "High": [x.high for x in b],
                    "Low": [x.low for x in b],
                    "Close": [x.close for x in b],
                    "Volume": [x.volume for x in b],
                }, index=pd.to_datetime([x.timestamp for x in b]))

                data_dict[symbol] = df.sort_index()
                print(f"{symbol}: {len(df)} bars")
            else:
                print(f"{symbol}: No data returned!!!")

    # Retrieve historical crypto data
    if crypto_symbols:
        request = CryptoBarsRequest(
            symbol_or_symbols=crypto_symbols,
            timeframe=timeframe,
            start=start_time,
            end=end_time
        )
        bars = CRYPTO_CLIENT.get_crypto_bars(request)

        for symbol in crypto_symbols:
            if symbol in bars.data and len(bars.data[symbol]) > 0:
                b = bars.data[symbol]
                df = pd.DataFrame({
                    "Open": [x.open for x in b],
                    "High": [x.high for x in b],
                    "Low": [x.low for x in b],
                    "Close": [x.close for x in b],
                    "Volume": [x.volume for x in b],
                }, index=pd.to_datetime([x.timestamp for x in b]))

                data_dict[symbol] = df.sort_index()
                print(f"{symbol}: {len(df)} bars")
            else:
                print(f"{symbol}: No data returned!!!")

    return data_dict

def calculate_basic_features(df):
    df = df.copy()

    returns = df["Close"].pct_change()

    df["returns"] = returns
    df["returns_5d"] = returns.rolling(5).sum()
    df["returns_10d"] = returns.rolling(10).sum()
    df["returns_20d"] = returns.rolling(20).sum()

    vol = returns.rolling(20).std()
    df["volatility_20d"] = vol
    df["volatility_60d"] = returns.rolling(60).std()

    for w in [5, 10, 20, 50, 100, 200]:
        ma = df["Close"].rolling(w).mean()
        df[f"ma_{w}"] = ma
        df[f"ma_ratio_{w}"] = df["Close"] / ma - 1

    df["momentum_20d"] = df["Close"] / df["Close"].shift(20) - 1

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    std = returns.rolling(20).std()
    ma20 = df["Close"].rolling(20).mean()
    upper = ma20 + 2 * std * ma20
    lower = ma20 - 2 * std * ma20
    df["bb_position"] = (df["Close"] - lower) / (upper - lower)

    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    roll_max = df["Close"].rolling(60).max()
    df["max_dd_60d"] = (df["Close"] - roll_max) / roll_max

    df["risk_adj_momentum_20d"] = df["returns_20d"] / (df["volatility_20d"] + 1e-6)

    return df

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_rolling_max_drawdown(prices, window):
    """Calculate rolling maximum drawdown"""
    def max_drawdown(price_series):
        running_max = price_series.expanding().max()
        drawdown = (price_series - running_max) / running_max
        return drawdown.min()
    
    return prices.rolling(window).apply(max_drawdown, raw=False)

def create_cross_asset_features(data_dict, market_symbol="SPY"):
    if market_symbol not in data_dict:
        print(f"{market_symbol} is missing hence skipping cross-asset features.")
        return data_dict

    market_returns = data_dict[market_symbol]["returns"]
    out = {}

    for sym, df in data_dict.items():
        df = df.copy()

        if sym != market_symbol and not CRYPTO_SYMBOLS(sym):
            cov = df["returns"].rolling(60).cov(market_returns)
            var = market_returns.rolling(60).var()
            df["beta"] = cov / var
            df["correlation_market"] = df["returns"].rolling(30).corr(market_returns)
            df["relative_strength"] = (
                df["Close"] / df["Close"].shift(20)
            ) / (data_dict[market_symbol]["Close"] / data_dict[market_symbol]["Close"].shift(20))
        else:
            df[["beta", "correlation_market", "relative_strength"]] = np.nan

        out[sym] = df

    return out


def prepare_ml_features(data_dict, target_forward_days=1, mode = "curated", for_prediction=False):
    """
    Prepare features for ML model training
    
    Args:
        data_dict: Dict of DataFrames with calculated features
        target_forward_days: Days forward for target calculation
    
    Returns:
        Combined DataFrame ready for ML training
    """
    feature_dfs = []
    
    for symbol, df in data_dict.items():
        # Select feature columns (exclude OHLCV and intermediate calculations)
        feature_cols = [col for col in df.columns if not any(x in col.lower() 
                       for x in ['open', 'high', 'low', 'close', 'volume', 'ma_', 'bb_upper', 'bb_lower'])]
        
        # TODO add more features as desired
        essential_features = ['returns', 'volatility_20d', 'ma_ratio_20',
                            'rsi', 'bb_position', 'volume_ratio', 'momentum_20d']
        
        # Filter to available features depending on the mode
        if mode == "curated":
            available_features = [col for col in essential_features if col in df.columns]
        elif mode == "all":
            # TODO experiment with all features
            available_features = feature_cols 

        if 'beta' in df.columns:
            available_features.extend(['beta', 'correlation_market', 'relative_strength'])
        
        # Create feature matrix for this symbol
        symbol_features = df[available_features].copy()
        symbol_features['symbol'] = symbol
        
        if not for_prediction:
            # Only create targets during training
            symbol_features['target_return'] = df['returns'].shift(-target_forward_days)
            symbol_features['target_volatility'] = df['volatility_20d'].shift(-target_forward_days)
        
        feature_dfs.append(symbol_features)
    
    combined_features = pd.concat(feature_dfs, ignore_index=False)

    if not for_prediction:
            # Drop rows where we don't have future targets
            pre_drop_count = len(combined_features)
            combined_features = combined_features.dropna(subset=['target_return', 'target_volatility'])
            post_drop_count = len(combined_features)
            
            rows_dropped = pre_drop_count - post_drop_count
            if rows_dropped > 0:
                print(f"  ⚠️  Dropped {rows_dropped} rows with unknown future targets")
                print(f"  Latest training data: {combined_features.index.max().date()}")


    # Drop NaN values in features as they could otherwise break model training
    combined_features = combined_features.dropna(subset=["returns", "volatility_20d", "momentum_20d", "rsi"])
   
    return combined_features

def get_market_data_for_ml(symbols, start_date, end_date=None):
    """
    Complete pipeline to fetch data and prepare features for ML
    
    Args:
        symbols: List of symbols to fetch
        start_date: Start date for data
        end_date: End date for data (None for today)
    
    Returns:
        DataFrame ready for ML model training
    """
    print("Starting market data pipeline for ML...")
    
    # Fetch historical data
    raw_data = fetch_historical_data(symbols, start_date, end_date)
    
    if not raw_data:
        print("No data fetched!!!")
        return pd.DataFrame()
    
    # Calculate features for each asset
    print("Calculating basic features...")
    featured_data = {}
    for symbol, df in raw_data.items():
        featured_data[symbol] = calculate_basic_features(df)
    
    # Add cross-asset features
    print("Adding cross-asset features...")
    enhanced_data = create_cross_asset_features(featured_data)
    
    # Prepare ML features
    print("Preparing ML features...")
    ml_features = prepare_ml_features(enhanced_data)
    
    print(f"Data pipeline complete: {ml_features.shape}")
    
    return ml_features

# Only for test purposes - this will not be invoked in the overall flow!!!
if __name__ == "__main__":
    # Test the data pipeline
    test_symbols = ["SPY", "GLD", "VXX", "BTC/USD","SOXX"]  # Subset for testing
    start_date = '2025-01-01'
    end_date = '2026-01-01'
    
    try:
        ml_data = get_market_data_for_ml(test_symbols, start_date)
        
        if not ml_data.empty:
            print("\n" + "="*50)
            print("SAMPLE DATA SUMMARY")
            print("="*50)
            print(f"Shape: {ml_data.shape}")
            print(f"Symbols: {ml_data['symbol'].unique()}")
            print(f"Date range: {ml_data.index[0]} to {ml_data.index[-1]}")
            print(f"Features: {[col for col in ml_data.columns if col not in ['symbol', 'target_return', 'target_volatility']]}")
            print("\nFirst few rows:")
            print(ml_data.head())
        else:
            print("No data generated")
            
    except Exception as e:
        print(f"Error in data pipeline: {e}")
        import traceback
        traceback.print_exc()
