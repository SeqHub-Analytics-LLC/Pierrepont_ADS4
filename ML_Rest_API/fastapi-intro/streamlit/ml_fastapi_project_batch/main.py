# app/main.py

from fastapi import FastAPI, File, UploadFile
from models import PredictionRequest, ModelType
from predict import predict_single, predict_batch
import pandas as pd
from io import StringIO

# Create the FastAPI app instance
app = FastAPI(
    title="Fitness ML Model API",
    description="A FastAPI-based REST service for predicting your bench press limit in Kilograms!",
    version="1.0.0"
)

# Route for the home page or info
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Fitness ML Model API!",
        "description": "This API predicts your bench press limit in kilograms based on input features.",
        "version": "1.0.0",
        "author": "Taiwo Togun",
        "model_info": {
            "models": "Random Forests, Decision Trees, Gradient Boosting", 
            "training_data": "Fitness-related data on exercises and body metrics"
        }
    }

@app.post("/predict")
def predict(data: PredictionRequest):
    prediction = predict_single(data)
    return {"prediction": prediction[0]}

@app.post("/predict_batch")
async def predict_batch_endpoint(file: UploadFile = File(...), model: ModelType = ModelType.rf):
    # Read the uploaded file into a pandas DataFrame
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))  #  CSV format for the file
    
    # Call the batch prediction function
    predictions = predict_batch(df, model)

    # Return predictions
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)