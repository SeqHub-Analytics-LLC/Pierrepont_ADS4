# app/predict.py
import joblib
import pandas as pd
from models import PredictionRequest
from utils import encode_features, scale_features, engineer_features
from fastapi import HTTPException

print("Loading models and encoders...")
# Load model and encoder
rf_model = joblib.load("artifacts/random_forests.pkl")
dtc_model = joblib.load("artifacts/decision_trees.pkl")
gradient_boosting = joblib.load("artifacts/gradient_boosting.pkl")


def predict_single(request: PredictionRequest):
    
    # Apply feature engineering, encoding and scaling
    df_results = engineer_features(request.Features)
    df_results = encode_features(df_results, encoder_path="artifacts/ordinal_encoder.pkl")
    df_results = scale_features(df_results, scaler_path="artifacts/minmax_scaler.pkl")
    

    print(df_results)
    #Restructure the columns; it must match the exact way the input were provided when the model was trained.
    columns = ['Sex', 'Equipment', 'BodyweightKg', 'BestSquatKg', 'BestDeadliftKg',
       'RelativeSquatStrength', 'RelativeDeadliftStrength', 'AgeCategory']
    df_restructured = df_results[columns]

    # Predict
    if request.model == "Random Forests":
      prediction = rf_model.predict(df_restructured)
    elif request.model == "Decision Trees":
      prediction = dtc_model.predict(df_restructured)
    elif request.model == "Gradient Boosting":
      prediction = gradient_boosting.predict(df_restructured)
    else:
      raise HTTPException(status_code=400, detail="Unsupported model type. Please choose 'Random Forests', 'Decision Trees', or 'Gradient Boosting'.")
    print(prediction)
    return prediction


