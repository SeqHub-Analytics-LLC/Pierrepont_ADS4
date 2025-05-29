from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List
import io
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = FastAPI(title="Injury Prediction API")
MODEL = joblib.load("artifacts/random_forest_model.pkl")
ENCODER = joblib.load("artifacts/oneHotEncoder.pkl")
SCALER = joblib.load("artifacts/standardScaler.pkl")
FEATURE_COLUMNS = joblib.load("artifacts/model_features.pkl")

class InputFeatures(BaseModel):
    Player_Weight: float
    Player_Height: float
    Previous_Injuries: str
    Position: str
    Training_Surface: str
    Player_Age: int
    Training_Intensity: float
    Recovery_Time: float

class PredictionRequest(BaseModel):
    Features: InputFeatures
    model: str = "Random Forests"

@app.get("/")
def root():
    return {"message": "Injury Prediction API is running"}

@app.post("/predict")
def predict_injury(request: PredictionRequest):
    try:
        input_data = request.Features.model_dump()
        df = pd.DataFrame([input_data])
        df = engineer_features(df)
        df = encode_categorical(df)
        df = scale_numerical(df)
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        prediction = MODEL.predict(df)
        return JSONResponse(content={"prediction": int(prediction[0])})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_probabilities")
async def predict_injury_probabilities(request: PredictionRequest):
    """Endpoint for injury probability prediction"""
    try:
        input_data = request.Features.model_dump()
        df = pd.DataFrame([input_data])
        df = engineer_features(df)
        df = encode_categorical(df)
        df = scale_numerical(df)
        probabilities = MODEL.predict_proba(df)
        return JSONResponse(content={
            "not_injured_prob": float(probabilities[0][0]),
            "injured_prob": float(probabilities[0][1])
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Endpoint for batch predictions from CSV"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        required_columns = [
            'Player_Weight', 'Player_Height', 'Previous_Injuries',
            'Position', 'Training_Surface', 'Player_Age',
            'Training_Intensity', 'Recovery_Time'
        ]
        
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain these columns: {required_columns}"
            )
        processed_df = process_dataframe(df)
        predictions = MODEL.predict(processed_df)
        probabilities = MODEL.predict_proba(processed_df)
        results = []
        for i in range(len(df)):
            results.append({
                "input_data": df.iloc[i].to_dict(),
                "prediction": int(predictions[i]),
                "probabilities": {
                    "not_injured_prob": float(probabilities[i][0]),
                    "injured_prob": float(probabilities[i][1])
                }
            })
        
        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
def engineer_features(df):
    """Apply feature engineering steps"""
    df['BMI'] = df['Player_Weight'] / (df['Player_Height'] / 100) ** 2
    gaps = [-float('inf'), 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
    categories = ['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III']
    df['BMI_Classification'] = pd.cut(df['BMI'], bins=gaps, labels=categories, right=False)
    df["Age_Group"] = pd.cut(
        df["Player_Age"],
        bins=[18, 22, 26, 30, 34, df["Player_Age"].max()],
        labels=["18-22", "23-26", "27-30", "31-34", "35+"],
        include_lowest=True,
    )
    return df

def encode_categorical(df):
    """Encode categorical features using pre-trained encoder"""
    df["Previous_Injuries"] = df["Previous_Injuries"].replace({"No": 0, "Yes": 1})
    categorical_cols = ["Position", "Training_Surface"]
    encoded = ENCODER.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=ENCODER.get_feature_names_out(categorical_cols), index=df.index)
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def scale_numerical(df):
    """Scale numerical features using pre-trained scaler"""
    numeric_cols = ["Player_Weight", "Player_Height", "Player_Age", 
                   "Training_Intensity", "Recovery_Time", "BMI"]
    df[numeric_cols] = SCALER.transform(df[numeric_cols])
    return df

def process_dataframe(df):
    """Process a complete dataframe through the pipeline"""
    df = engineer_features(df)
    df = encode_categorical(df)
    df = scale_numerical(df)
    return df

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)