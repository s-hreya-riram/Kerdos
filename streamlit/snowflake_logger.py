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
            # Handle None/NaN FIRST
            if val is None or pd.isna(val):
                return None
            
            # Check for float NaN explicitly
            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    return None
            
            # Check for numpy NaN
            if isinstance(val, (np.floating,)):
                if np.isnan(val) or np.isinf(val):
                    return None

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

            # numpy numeric → python numeric
            if isinstance(val, (np.integer,)):
                return int(val)

            if isinstance(val, (np.floating,)):
                return float(val)
            
            # numpy bool → python bool
            if isinstance(val, (np.bool_,)):
                return bool(val)

            return val

        df = df.map(normalize_value)
        
        # STEP 3: Filter out rows with NaN in critical columns
        # For predictions table, we need valid SYMBOL at minimum
        if table_name == "STRATEGY_PREDICTIONS":
            # Keep rows where SYMBOL is not None
            df = df[df['SYMBOL'].notna()]
            # Convert NaN predictions to None is already done above
            
        # STEP 4: Debug logging
        initial_rows = len(df)
        
        # Remove any rows that still have string 'nan' or 'NaN'
        for col in df.columns:
            if df[col].dtype == 'object':
                df = df[~df[col].astype(str).str.upper().isin(['NAN', 'NONE', 'NAT'])]
        
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
            # Convert to list and insert
            values_list = df.values.tolist()
            
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
    # Convenience Upload Helpers
    # ---------------------------------------------------

    def upload_performance(self, df: pd.DataFrame):
        self.upload_dataframe(df, "STRATEGY_PERFORMANCE")

    def upload_positions(self, df: pd.DataFrame):
        self.upload_dataframe(df, "STRATEGY_POSITIONS")

    def upload_predictions(self, df: pd.DataFrame):
        self.upload_dataframe(df, "STRATEGY_PREDICTIONS")

    # ---------------------------------------------------

    def close(self):
        self.conn.close()