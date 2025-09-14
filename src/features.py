import numpy as np

def add_lagged_features(df, col, lags=[1,2]):
    """Add lagged versions of a column (weekly effects)."""
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def log_transform(df, cols):
    """Apply log1p to skewed spend variables."""
    for c in cols:
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(df[c])
    return df
