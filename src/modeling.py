import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_revenue_model(df: pd.DataFrame):
    """
    Stage 2: Train revenue model using Google_spend_pred + other levers.
    """
    feature_cols = [
        "google_spend_pred", "facebook_spend", "tiktok_spend",
        "instagram_spend", "snapchat_spend", "social_followers",
        "average_price", "promotions", "emails_send", "sms_send"
    ]

    X = df[feature_cols]
    y = df["revenue"]

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)

    return model, mse


