import pandas as pd

def to_utc(ts):
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
