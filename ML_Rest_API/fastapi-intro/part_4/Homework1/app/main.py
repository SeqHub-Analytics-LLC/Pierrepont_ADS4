# app/main.py

from fastapi import FastAPI
from models import PredictionRequest
from predict import predict_single

# Create the FastAPI app instance
app = FastAPI(
    title="Sports Injury ML Model API",
    description="A FastAPI-based REST service for predicting your bench press limit in Kilograms!",
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
    prediction = predict_single(data)
    return {"prediction": prediction[0]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)