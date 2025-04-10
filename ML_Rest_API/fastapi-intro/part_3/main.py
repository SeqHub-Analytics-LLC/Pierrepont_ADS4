import pandas as pd
from preprocessing import (
    feature_engineering,
    split_data,
    encode_features,
    scale_features
)
from train_model import run_optimization, evaluate_model
from config import DATA_PATH

def main():
    df = pd.read_csv(DATA_PATH)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = encode_features(X_train, X_test)
    X_train, X_test = scale_features(X_train, X_test)
    model = run_optimization(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()