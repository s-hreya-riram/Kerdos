import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
import math

load_dotenv()


class SnowflakeUploader:

    def __init__(self):

        self.conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PRODUCTION")
        )
        
        cursor = self.conn.cursor()
        cursor.execute(f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}")
        cursor.execute(f"USE DATABASE {os.getenv('SNOWFLAKE_DATABASE')}")
        cursor.execute(f"USE SCHEMA {os.getenv('SNOWFLAKE_SCHEMA', 'PRODUCTION')}")
        cursor.close()

        self._create_tables()

    # ---------------------------------------------------
    # Table Creation
    # ---------------------------------------------------

    def _create_tables(self):

        cursor = self.conn.cursor()

        # ---- Performance Table ----
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS STRATEGY_PERFORMANCE (
            STRATEGY_NAME STRING,
            TIMESTAMP TIMESTAMP_NTZ,
            PORTFOLIO_VALUE FLOAT,
            CASH FLOAT,
            IS_OUT_OF_SAMPLE BOOLEAN
        )
        """)

        # ---- Positions Table ----
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS STRATEGY_POSITIONS (
            STRATEGY_NAME STRING,
            TIMESTAMP TIMESTAMP_NTZ,
            SYMBOL STRING,
            QUANTITY FLOAT,
            MARKET_VALUE FLOAT,
            AVG_PRICE FLOAT
        )
        """)

        # ---- Predictions Table (ML only but harmless for benchmarks) ----
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS STRATEGY_PREDICTIONS (
            STRATEGY_NAME STRING,
            TIMESTAMP TIMESTAMP_NTZ,
            SYMBOL STRING,
            PREDICTED_RETURN FLOAT,
            PREDICTED_VOL FLOAT
        )
        """)

        # ---- Competition Performance Table ----
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS COMPETITION_PERFORMANCE (
            STRATEGY_NAME STRING,
            TIMESTAMP TIMESTAMP_NTZ,
            PORTFOLIO_VALUE FLOAT,
            CASH FLOAT
        )
        """)

        # ---- Competition Positions Table ----
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS COMPETITION_POSITIONS (
            STRATEGY_NAME STRING,
            TIMESTAMP TIMESTAMP_NTZ,
            SYMBOL STRING,
            QUANTITY FLOAT,
            MARKET_VALUE FLOAT,
            AVG_PRICE FLOAT
        )
        """)

        cursor.close()

    # ---------------------------------------------------
    # Generic Upload with Complete NaN Handling
    # ---------------------------------------------------
    def upload_dataframe(self, df: pd.DataFrame, table_name: str):
        if df.empty:
            return

        df = df.copy()
        
        # STEP 1: Replace all NaN/inf with None BEFORE type conversion
        df = df.replace([np.inf, -np.inf], None)
        
        # For numeric columns, explicitly convert NaN to None
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                df[col] = df[col].where(pd.notna(df[col]), None)

        # STEP 2: Convert types to Python/string format
        def normalize_value(val):
            # Handle None/NaN FIRST - be more aggressive
            if val is None:
                return None
            
            # Check for pandas NaN
            if pd.isna(val):
                return None
            
            # Check for numpy NaN
            if isinstance(val, (np.floating, np.integer)) and (np.isnan(val) if hasattr(np, 'isnan') and not isinstance(val, np.integer) else False):
                return None
                
            # Check for float NaN explicitly
            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    return None
            
            # Check for string representations of NaN
            if isinstance(val, str) and val.upper() in ['NAN', 'NONE', 'NAT', '', 'NULL']:
                return None
            
            # Convert the special case where Python shows 'nan' as a value
            try:
                if str(val).lower() in ['nan', 'none', 'nat']:
                    return None
            except:
                pass

            # Convert timestamps to STRING format
            if isinstance(val, pd.Timestamp):
                try:
                    return val.tz_localize(None).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    return val.strftime('%Y-%m-%d %H:%M:%S')

            # numpy datetime64 → string
            if isinstance(val, (np.datetime64,)):
                ts = pd.Timestamp(val)
                return ts.strftime('%Y-%m-%d %H:%M:%S')
            
            # Python datetime → string
            if isinstance(val, datetime):
                if val.tzinfo is not None:
                    val = val.replace(tzinfo=None)
                return val.strftime('%Y-%m-%d %H:%M:%S')

            # numpy numeric → python numeric (with NaN check)
            if isinstance(val, (np.integer,)):
                return int(val)

            if isinstance(val, (np.floating,)):
                # Double-check for NaN in numpy floats
                if np.isnan(val):
                    return None
                return float(val)
            
            # numpy bool → python bool
            if isinstance(val, (np.bool_,)):
                return bool(val)

            return val

        df = df.map(normalize_value)
        
        # STEP 3: More aggressive NaN filtering
        # Remove rows where critical columns have None/NaN
        if table_name == "STRATEGY_PREDICTIONS":
            # Keep rows where SYMBOL is not None AND at least one prediction is not None
            df = df[df['SYMBOL'].notna()]
            
            # For predictions, if both PREDICTED_RETURN and PREDICTED_VOL are None, skip the row
            df = df[~((df['PREDICTED_RETURN'].isna()) & (df['PREDICTED_VOL'].isna()))]
            
            # Replace remaining NaN with None explicitly
            df['PREDICTED_RETURN'] = df['PREDICTED_RETURN'].where(pd.notna(df['PREDICTED_RETURN']), None)
            df['PREDICTED_VOL'] = df['PREDICTED_VOL'].where(pd.notna(df['PREDICTED_VOL']), None)
        
        # STEP 4: Final cleanup - remove any rows that still contain string representations of NaN
        initial_rows = len(df)
        
        # Remove any rows that still have string 'nan', 'NaN', etc.
        for col in df.columns:
            if df[col].dtype == 'object':
                # Remove rows where string values are NaN representations
                mask = ~df[col].astype(str).str.upper().isin(['NAN', 'NONE', 'NAT', 'NULL', ''])
                df = df[mask]
        
        # Also check for actual NaN values that might have slipped through
        # For predictions table, remove any rows with NaN in prediction columns
        if table_name == "STRATEGY_PREDICTIONS":
            nan_mask = df['PREDICTED_RETURN'].isna() & df['PREDICTED_VOL'].isna()
            df = df[~nan_mask]
        
        final_rows = len(df)
        
        if initial_rows != final_rows:
            print(f"   ⚠️  Filtered {initial_rows - final_rows} rows with NaN values from {table_name}")
        
        if df.empty:
            print(f"   ⚠️  No valid rows to insert into {table_name}")
            return

        cursor = self.conn.cursor()

        columns = ", ".join(df.columns)
        placeholders = ", ".join(["%s"] * len(df.columns))

        insert_sql = f"""
            INSERT INTO {table_name} ({columns})
            VALUES ({placeholders})
        """

        try:
            # Convert to list and do final NaN check
            values_list = []
            for _, row in df.iterrows():
                row_values = []
                for val in row:
                    # Final safety check before inserting
                    if pd.isna(val) or (isinstance(val, str) and val.upper() in ['NAN', 'NONE', 'NAT']):
                        row_values.append(None)
                    else:
                        row_values.append(val)
                values_list.append(row_values)
            
            # Final sanity check - print first row for debugging
            if values_list:
                print(f"   📝 Inserting {len(values_list)} rows into {table_name}")
                
            cursor.executemany(insert_sql, values_list)
            self.conn.commit()
            cursor.close()
            
        except Exception as e:
            print(f"❌ Error inserting into {table_name}: {e}")
            print(f"   Total rows attempted: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample row (first):")
            print(f"   {df.iloc[0].to_dict()}")
            if len(df) > 1:
                print(f"   Sample row (last):")
                print(f"   {df.iloc[-1].to_dict()}")
            
            # Enhanced debugging - check the actual values going to Snowflake
            print(f"   📋 First processed row for insertion: {values_list[0] if values_list else 'EMPTY'}")
            if len(values_list) > 1:
                print(f"   📋 Last processed row for insertion: {values_list[-1]}")
            
            # Check for any NaN in the actual values
            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                for k, v in row_dict.items():
                    if isinstance(v, str) and v.upper() in ['NAN', 'NONE', 'NAT']:
                        print(f"   🔴 Found string NaN in row {idx}, column {k}: {v}")
                    if pd.isna(v):
                        print(f"   🔴 Found pd.isna in row {idx}, column {k}")
                        
            cursor.close()
            raise


    # ---------------------------------------------------
    # Convenience Upload Helpers - Historical Tables
    # ---------------------------------------------------

    def upload_performance(self, df: pd.DataFrame):
        self.upload_dataframe(df, "STRATEGY_PERFORMANCE")

    def upload_positions(self, df: pd.DataFrame):
        self.upload_dataframe(df, "STRATEGY_POSITIONS")

    def upload_predictions(self, df: pd.DataFrame):
        self.upload_dataframe(df, "STRATEGY_PREDICTIONS")

    # ---------------------------------------------------
    # Convenience Upload Helpers - Competition Tables
    # ---------------------------------------------------

    def upload_competition_performance(self, df: pd.DataFrame):
        """Upload performance data to COMPETITION_PERFORMANCE table"""
        # Ensure only competition columns are present
        required_cols = ['STRATEGY_NAME', 'TIMESTAMP', 'PORTFOLIO_VALUE', 'CASH']
        df = df[required_cols].copy()
        self.upload_dataframe(df, "COMPETITION_PERFORMANCE")

    def upload_competition_positions(self, df: pd.DataFrame):
        """Upload positions data to COMPETITION_POSITIONS table"""
        # Ensure only competition columns are present
        required_cols = ['STRATEGY_NAME', 'TIMESTAMP', 'SYMBOL', 
                        'QUANTITY', 'MARKET_VALUE', 'AVG_PRICE']
        df = df[required_cols].copy()
        self.upload_dataframe(df, "COMPETITION_POSITIONS")

    # ---------------------------------------------------

    def close(self):
        self.conn.close()