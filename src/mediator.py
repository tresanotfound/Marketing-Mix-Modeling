import pandas as pd
from sklearn.linear_model import LinearRegression

def stage1_google_model(df: pd.DataFrame):
    """
    Stage 1: Model Google spend as a function of Facebook, TikTok, Instagram, Snapchat.
    """
    feature_cols = ["facebook_spend", "tiktok_spend", "instagram_spend", "snapchat_spend"]
    X = df[feature_cols]
    y = df["google_spend"]

    model = LinearRegression()
    model.fit(X, y)

    # add predicted google spend
    df["google_spend_pred"] = model.predict(X)

    return model, df

