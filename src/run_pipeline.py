from src.data_prep import load_data, preprocess
from src.mediator import train_google_mediator
from src.modeling import train_revenue_model
from src.evaluation import plot_predictions

def main():
    # Load & clean
    df = load_data()
    df = preprocess(df)

    # Stage 1: Google mediator
    mediator_model, df = train_google_mediator(df)

    # Stage 2: Revenue model
    rev_model, mse, r2 = train_revenue_model(df)
    print(f"Revenue Model -> MSE: {mse:.2f}, R2: {r2:.3f}")

    # Plot predictions
    feature_cols = [
        "google_spend_pred", "average_price", "promotions",
        "emails_send", "sms_send", "social_followers"
    ]
    plot_predictions(df, rev_model, feature_cols)

if __name__ == "__main__":
    main()
