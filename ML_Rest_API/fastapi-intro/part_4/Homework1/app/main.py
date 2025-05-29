# app/main.py

from fastapi import FastAPI, HTTPException
from models import PredictionRequest
from predict import predict_single, predict_probabilities
from typing import List

# Create the FastAPI app instance
app = FastAPI(
    title="Sports Injury ML Model API",
    description="A FastAPI-based REST service for predicting whether a player has an injury!",
    version="1.0.0"
)

# Route for the home page or info
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Sports Injury ML Model API!",
        "description": "This API predicts your likelihood of being injured based on input features.",
        "version": "1.0.0",
        "author": "Taiwo Togun",
        "model_info": {
            "models": "Random Forests", 
            "training_data": "Training-related data in soccer and body metrics"
        }
    }

@app.post("/predict")
def predict(data: PredictionRequest):
    #print(data)
    prediction = int(list(predict_single(data))[0])   
    return {"prediction": prediction}


@app.post("/predict_probabilities")
def predict_probabilities_endpoint(data: PredictionRequest):
    try:
        probabilities = predict_probabilities(data)
        return {"probabilities": probabilities}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
def predict_batch(data: List[PredictionRequest]):
    try:
        predictions = [int(list(predict_single(item))[0]) for item in data]
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)