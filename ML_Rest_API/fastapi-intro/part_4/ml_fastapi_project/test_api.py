# test_api.py

import pytest
from fastapi.testclient import TestClient
from main import app
import pandas as pd
from io import StringIO

client = TestClient(app)

# Sample input data for the /predict endpoint
sample_single_input = {
    "Features": {
        "Sex": "Male",
        "Equipment": "Wraps",
        "Age": 25,
        "BodyweightKg": 80,
        "BestSquatKg": 150,
        "BestDeadliftKg": 180
    },
    "model": "Random Forests"
}

# Test /predict endpoint
def test_predict():
    # Send POST request to /predict endpoint with the sample data
    response = client.post("/predict", json=sample_single_input)
    
    # Ensure status code is 200 OK
    assert response.status_code == 200
    
    # Assert the structure of the response
    assert "prediction" in response.json()
    
    # Optionally, assert if the prediction is of numeric type (depending on your model's output)
    prediction = response.json()["prediction"]
    assert isinstance(prediction, (int, float))


# Test invalid model type in /predict endpoint
def test_invalid_model_predict():
    invalid_input = sample_single_input.copy()
    invalid_input["model"] = "Invalid Model"
    
    response = client.post("/predict", json=invalid_input)
    
    # Ensure status code is 422 for invalid model type
    assert response.status_code == 422
    print(response.json())

if __name__ == "__main__":
    print("Test: test_predict()")
    test_predict()
    print("Test: test_invalid_model_predict")
    test_invalid_model_predict()