import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from preprocessing import feature_engineering, encode_features, scale_features
from train_model import train_model,evaluate_model
from model_utils import save_pickle
from config import one_hot_cols, scale_cols, MODEL_PATH

mlflow.end_run()  # End any previous runs
mlflow.set_experiment("sports_injury_prediction1")


df = pd.read_csv("data/injury_data_with_categories.csv")
df = feature_engineering(df)

X = df.drop(columns=["Likelihood_of_Injury"])
y = df["Likelihood_of_Injury"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoding and scaling
X_train_encoded, X_test_encoded, _ = encode_features(X_train, X_test, one_hot_cols)
X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test, scale_cols)


with mlflow.start_run():
    
    model = train_model(X_train_scaled, y_train)

    evaluate_model(model, X_test_scaled, y_test)

    save_pickle(model, MODEL_PATH)
