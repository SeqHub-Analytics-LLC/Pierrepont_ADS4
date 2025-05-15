# app/predict.py
import joblib
import pandas as pd
from models import PredictionRequest
from model_utils import encode_features, scale_features, engineer_features
from fastapi import HTTPException


# Load model and encoder
rf_model = joblib.load("artifacts/random_forests.pkl")


def predict_single(request: PredictionRequest):

  df = pd.DataFrame([request.f.model_dump() for f in request.Features])
  print(df)
  # Apply feature engineering, encoding and scaling
  df_results = engineer_features(df)
  df_results = encode_features(df_results, categorical_cols=["Position", "Training_Surface" ],  path="artifacts/oneHotEncoder.pkl")
  df_results = scale_features(df_results, numeric_cols=["Player_Weight", "Player_Height", "Player_Age", "Training_Intensity", "Recovery_Time"] ,path="artifacts/standardScaler.pkl")
  

  #print(df_results)
  #prediction = rf_model.predict(df_results)
  #return prediction

if __name__ == "__main__":
    # Example usage
    print("Predicting for example request...")
    example_request = PredictionRequest(
      Features={
        "Player_Weight": 75.0,
        "Player_Height": 180.0,
        "Previous_Injuries": "No",
        "Position": "Midfielder",
        "Training_Surface": "Artificial Turf",
        "Player_Age": 25,
        "Training_Intensity": 0.8,
        "Recovery_Time": 1.5
      },
      model="Random Forests"
    )
    
    prediction = predict_single(example_request)
    print(f"Prediction: {prediction}")