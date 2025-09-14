import pandas as pd

def load_data(path="data/Assessment2_MMMWeekly.csv"):
    """Load dataset and parse week column."""
    df = pd.read_csv(path)
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]
    if "week" in df.columns:
        df["week"] = pd.to_datetime(df["week"])
        df = df.sort_values("week")
    return df

def preprocess(df):
    """Handle missing values and zero spends."""
    df = df.fillna(0)
    # Replace negative or nonsensical spends/revenue with 0
    spend_cols = [c for c in df.columns if "spend" in c]
    for col in spend_cols + ["revenue"]:
        df[col] = df[col].clip(lower=0)
    return df
