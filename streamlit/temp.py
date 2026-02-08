import pandas as pd
import snowflake.connector
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

print("SNOWFLAKE_ACCOUNT =", os.getenv("SNOWFLAKE_ACCOUNT"))

conn = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASSWORD'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
    database="KERDOS_FUND",
    schema="KERDOS_SCHEMA",
    role="ACCOUNTADMIN"
)

df = pd.read_sql("SELECT * FROM STRATEGY_PERFORMANCE LIMIT 5", conn)
print(df)
