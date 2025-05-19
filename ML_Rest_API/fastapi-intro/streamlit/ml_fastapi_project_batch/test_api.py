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
        "Equipment": "Barbell",
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

# Test /predict_batch endpoint
def test_predict_batch():
    # Create a sample CSV file as input data
    data = {
        "Sex": ["Male", "Female"],
        "Equipment": ["Barbell", "Dumbbell"],
        "BodyweightKg": [80, 55],
        "BestSquatKg": [150, 100],
        "BestDeadliftKg": [180, 120],
        "RelativeSquatStrength": [1.875, 1.818],
        "RelativeDeadliftStrength": [2.25, 2.182],
        "AgeCategory": ["Open", "Junior"]
    }

    df = pd.DataFrame(data)
    csv_data = df.to_csv(index=False)

    # Send POST request to /predict_batch endpoint with the file
    response = client.post(
        "/predict_batch", 
        files={"file": ("test_data.csv", StringIO(csv_data), "text/csv")},
        params={"model": "Random Forests"}
    )
    
    # Ensure status code is 200 OK
    assert response.status_code == 200
    
    # Assert that predictions are returned
    assert "predictions" in response.json()
    
    # Check if predictions is a list (depending on your model output)
    predictions = response.json()["predictions"]
    assert isinstance(predictions, list)
    
    # Optionally, check that the number of predictions matches the number of rows in the input data
    assert len(predictions) == len(df)

# Test invalid model type in /predict endpoint
def test_invalid_model_predict():
    invalid_input = sample_single_input.copy()
    invalid_input["model"] = "Invalid Model"
    
    response = client.post("/predict", json=invalid_input)
    
    # Ensure status code is 422 for invalid model type
    assert response.status_code == 422
    print(response.json())

# Test missing columns in /predict_batch
def test_missing_columns_predict_batch():
    # Create a sample CSV file with missing a required column (e.g., 'RelativeSquatStrength')
    data = {
        "Sex": ["Male"],
        "Equipment": ["Barbell"],
        "BodyweightKg": [80],
        "BestSquatKg": [150],
        "BestDeadliftKg": [180],
        "AgeCategory": ["Open"]
    }
    
    df = pd.DataFrame(data)
    csv_data = df.to_csv(index=False)

    # Send POST request to /predict_batch with missing columns
    response = client.post(
        "/predict_batch", 
        files={"file": ("test_data.csv", StringIO(csv_data), "text/csv")},
        params={"model": "Random Forests"}
    )
    
    # Ensure status code is 422 and error message is returned
    assert response.status_code == 422
    print(response.json())


if __name__ == "__main__":
    print("Test: test_predict()")
    test_predict()
    print("Test: test_invalid_model_predict")
    test_invalid_model_predict()
    print("Test: test_predict_batch()")
    test_predict_batch()
    print("Test: test_missing_columns_predict_batch")
    test_missing_columns_predict_batch()
