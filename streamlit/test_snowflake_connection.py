"""
Snowflake Connection Diagnostic
Tests connection and warehouse activation
"""

import os
import sys
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

def test_connection():
    """Test Snowflake connection step by step"""
    
    print("="*60)
    print("SNOWFLAKE CONNECTION DIAGNOSTIC")
    print("="*60)
    
    # Step 1: Check environment variables
    print("\n1️⃣ Checking environment variables...")
    required_vars = [
        'SNOWFLAKE_USER',
        'SNOWFLAKE_PASSWORD', 
        'SNOWFLAKE_ACCOUNT',
        'SNOWFLAKE_WAREHOUSE',
        'SNOWFLAKE_DATABASE',
        'SNOWFLAKE_SCHEMA'
    ]
    
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask password
            display = '*' * len(value) if 'PASSWORD' in var else value
            print(f"   ✅ {var} = {display}")
        else:
            print(f"   ❌ {var} = NOT SET")
            missing.append(var)
    
    if missing:
        print(f"\n❌ Missing variables: {missing}")
        return False
    
    # Step 2: Test basic connection
    print("\n2️⃣ Testing basic connection...")
    try:
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PRODUCTION")
        )
        print("   ✅ Connected to Snowflake")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    # Step 3: Test warehouse activation
    print("\n3️⃣ Testing warehouse activation...")
    try:
        cursor = conn.cursor()
        
        warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
        print(f"   Activating warehouse: {warehouse}")
        cursor.execute(f"USE WAREHOUSE {warehouse}")
        print("   ✅ Warehouse activated")
        
        # Verify it's active
        cursor.execute("SELECT CURRENT_WAREHOUSE()")
        active_wh = cursor.fetchone()[0]
        print(f"   ✅ Current warehouse: {active_wh}")
        
        cursor.close()
    except Exception as e:
        print(f"   ❌ Warehouse activation failed: {e}")
        conn.close()
        return False
    
    # Step 4: Test database/schema
    print("\n4️⃣ Testing database and schema...")
    try:
        cursor = conn.cursor()
        
        database = os.getenv('SNOWFLAKE_DATABASE')
        schema = os.getenv('SNOWFLAKE_SCHEMA', 'PRODUCTION')
        
        cursor.execute(f"USE DATABASE {database}")
        print(f"   ✅ Using database: {database}")
        
        cursor.execute(f"USE SCHEMA {schema}")
        print(f"   ✅ Using schema: {schema}")
        
        # Verify
        cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
        db, sch = cursor.fetchone()
        print(f"   ✅ Current database: {db}")
        print(f"   ✅ Current schema: {sch}")
        
        cursor.close()
    except Exception as e:
        print(f"   ❌ Database/schema setup failed: {e}")
        conn.close()
        return False
    
    # Step 5: Test table access
    print("\n5️⃣ Testing table access...")
    try:
        cursor = conn.cursor()
        
        # Try to query existing tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"   ✅ Found {len(tables)} tables")
        for table in tables:
            print(f"      - {table[1]}")  # Table name
        
        # Try to query STRATEGY_PERFORMANCE
        cursor.execute("SELECT COUNT(*) FROM STRATEGY_PERFORMANCE")
        count = cursor.fetchone()[0]
        print(f"   ✅ STRATEGY_PERFORMANCE has {count} rows")
        
        cursor.close()
    except Exception as e:
        print(f"   ⚠️  Table access issue: {e}")
        # Not critical - tables might not exist yet
    
    # Step 6: Test insert
    print("\n6️⃣ Testing insert...")
    try:
        cursor = conn.cursor()
        
        test_data = {
            'STRATEGY_NAME': 'TEST',
            'TIMESTAMP': '2024-01-01 00:00:00',
            'PORTFOLIO_VALUE': 10000.0,
            'CASH': 1000.0,
            'IS_OUT_OF_SAMPLE': False
        }
        
        cursor.execute("""
            INSERT INTO STRATEGY_PERFORMANCE 
            (STRATEGY_NAME, TIMESTAMP, PORTFOLIO_VALUE, CASH, IS_OUT_OF_SAMPLE)
            VALUES (%s, %s, %s, %s, %s)
        """, list(test_data.values()))
        
        conn.commit()
        print("   ✅ Test insert successful")
        
        # Clean up test data
        cursor.execute("DELETE FROM STRATEGY_PERFORMANCE WHERE STRATEGY_NAME = 'TEST'")
        conn.commit()
        print("   ✅ Test data cleaned up")
        
        cursor.close()
    except Exception as e:
        print(f"   ❌ Insert test failed: {e}")
        conn.close()
        return False
    
    # Cleanup
    conn.close()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)