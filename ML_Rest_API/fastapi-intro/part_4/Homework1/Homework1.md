# Serving the Injury Prediction Model with FastAPI

In this assignment, you are expected to **leverage your previous work** where you converted a Jupyter notebook into modular Python scripts, and now your task is to **serve the injury prediction model using FastAPI**. The primary goal is to build two API endpoints:

1. **Prediction Endpoint**: Predict the class (injured vs. not injured).
2. **Probability Endpoint**: Predict the probabilities of injury (class probability).

---

## 📌 Objective

The goal of this project is to **serve** an already trained injury prediction model using **FastAPI**. You will build two API endpoints for this:

- **`/predict`**: Accepts input features and returns the predicted class (e.g., injured or not injured).
- **`/predict_probabilities`**: Accepts input features and returns the predicted probabilities (e.g., the likelihood of being injured).

The assignment reinforces the process of:
- **Serving pre-trained models** using FastAPI.
- Creating API endpoints for both class predictions and probability predictions.

---

## 🛠️ Task Breakdown

You are **not required to train the model again**. Instead, you will leverage your previously trained model and the scripts you’ve created to deploy it through FastAPI. Follow these steps:

- **Step 1**: Use the **saved model, encoder, and scaler** from the previous assignment.
- **Step 2**: **Build a FastAPI app** that serves the pre-trained model:
  - **`/predict`**: Class prediction endpoint.
  - **`/predict_probabilities`**: Probability prediction endpoint.
- **Step 3**: Structure the FastAPI project with clean, modular code, ensuring separation of concerns and maintainability.

---

### 🗂️ Recommended Project Structure

```
Homework1/
├── app/
│   ├── artifacts/
│   │   ├── random_forests.pkl         # Pre-trained model
│   │   ├── ordinal_encoder.pkl        # Encoder for categorical features
│   │   └── minmax_scaler.pkl          # Scaler for numerical features
│   ├── main.py                        # FastAPI app with endpoints for predictions
│   ├── predict.py                     # Logic for loading models and making predictions
│   ├── preprocessing.py               # Data cleaning, feature engineering, scaling
│   ├── config.py                      # Configuration file for constants and paths
│   ├── model_utils.py                 # Functions to load and use model, encoder, and scaler
│   └── requirements.txt               # Required Python dependencies
└── README.md                          # This file with project instructions
```

---

## ✍️ Submission Instructions

1. **Push All Files**: Make sure all files are placed under the `homework1/` directory.

2. **Run the FastAPI App**: You should be able to start the FastAPI app by running the following:
   ```bash
   python main.py
   ```
   Your API should be accessible at `http://127.0.0.1:8000`.

3. **Test Endpoints**: The FastAPI app will automatically generate documentation at `http://127.0.0.1:8000/docs`. Use this interactive UI to test the `/predict` and `/predict_probabilities` endpoints.

---

## 📎 Reference

- Previous Assignment (Jupyter Notebook to Python Script Conversion): [Homework 1](#)
- FastAPI Documentation: [FastAPI Docs](https://fastapi.tiangolo.com/)


### Good luck building and deploying your injury prediction API with FastAPI! 🚀 
