import joblib
import pandas as pd
from processing import load_and_clean_data, create_features, encode_features, scale_features

def make_prediction(path, test_file):
    """Loads a saved model and evaluates it on new test data."""
    model = joblib.load(path + "random_forest_model.pkl")
    #load data
    test_data = load_and_clean_data(test_file)
    test_data = create_features(test_data)
    #encode data
    test_data, _ = encode_features(test_data, use_saved=True)
    #scale the data
    test_scaled = scale_features(X_train = None, X_test = test_data, use_saved=True)
    predictions = model.predict(test_scaled)
    print("Predictions Generated!")
    return predictions

if __name__ == "__main__":
    make_prediction(path = "data/artifacts/", test_file = "data/test_data.csv")
 
