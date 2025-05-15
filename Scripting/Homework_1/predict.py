import joblib
import pandas as pd
from data_processing import scale_features, load_and_clean_data

def make_prediction(model_path, test_file):
    """Loads a saved model and evaluates it on new test data."""
    model = joblib.load(model_path)
    test_data = load_and_clean_data(test_file)
    X_scaled, _ = scale_features(test_data)  # Assume scaler is already saved elsewhere
    predictions = model.predict(X_scaled)
    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    make_prediction("model_output/random_forest_model.pkl", "test_data.csv")