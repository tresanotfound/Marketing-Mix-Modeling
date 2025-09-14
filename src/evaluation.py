import matplotlib.pyplot as plt

def plot_predictions(df, model, features=None):
    """
    Plot actual vs predicted revenue.
    """
    if features is None:
        features = [
            "google_spend_pred", "facebook_spend", "tiktok_spend",
            "instagram_spend", "snapchat_spend", "social_followers",
            "average_price", "promotions", "emails_send", "sms_send"
        ]

    preds = model.predict(df[features])

    plt.figure(figsize=(10, 6))
    plt.plot(df["week"], df["revenue"], label="Actual Revenue", marker="o")
    plt.plot(df["week"], preds, label="Predicted Revenue", marker="x")
    plt.xlabel("Week")
    plt.ylabel("Revenue")
    plt.title("Actual vs Predicted Revenue")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
