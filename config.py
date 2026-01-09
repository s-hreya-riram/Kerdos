import os
from dotenv import load_dotenv

# Load environment variables from the .env file (if present)
load_dotenv()

ALPACA_CONFIG = {
    "API_KEY": os.getenv("ALPACA_API_KEY"),
    "API_SECRET": os.getenv("ALPACA_API_SECRET"),
    "PAPER": True,
}