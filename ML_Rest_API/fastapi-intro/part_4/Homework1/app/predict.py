# app/predict.py
import joblib
import pandas as pd
from models import PredictionRequest
from model_utils import encode_features, scale_features, engineer_features
from fastapi import HTTPException


# Load model and encoder
rf_model = joblib.load("artifacts/random_forests.pkl")


def predict_single(request: PredictionRequest):
  # Apply feature engineering, encoding and scaling
  df_results = engineer_features(request.Features)
  df_results = encode_features(df_results, encoder_path="artifacts/ordinal_encoder.pkl")
  df_results = scale_features(df_results, scaler_path="artifacts/minmax_scaler.pkl")
  
  #Restructure the columns; it must match the exact way the input were provided when the model was trained.
  columns = ["Player_Weight",
    "Player_Height",
    "Previous_Injuries",
    "Position",
    "Training_Surface",
    "Player_Age",
    "Training_Intensity",
    "Recovery_Time"]
  df_restructured = df_results[columns]

  # Predict
  prediction = rf_model.predict(df_restructured)
  print(prediction)
  return prediction
