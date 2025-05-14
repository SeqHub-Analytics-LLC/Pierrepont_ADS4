import streamlit as st
from PIL import Image
import requests

# Title of the app
st.title("🏋️‍♂️ Bench Press Limit Prediction")

# Sidebar for selecting prediction type (no default selected)
prediction_type = st.sidebar.radio("Choose Prediction Type", 
                                  ["Single-point Prediction", "Batch Prediction"],
                                  index=None)  # index=0 to ensure the first item is blank

# Home Page (Welcome Message) - Displayed if no prediction type is selected
if prediction_type == None:
    # Display the introduction page when no prediction type is selected
    st.markdown("""
        **Welcome to the Bench Press Strength Tracker!**
        
        This application is designed to help you track and predict your bench press strength. Whether you're training for strength or just tracking your progress, this tool uses advanced algorithms to predict your maximum bench press limit based on your current performance.
    """)

    image = Image.open("artifacts/benchpress.webp")  # Make sure to put the correct path to your image
    st.image(image, caption="Track and Predict Your Bench Press Strength!", use_container_width=True)

    st.markdown("""
        **How It Works:**
        - You provide your **weight lifted** (in kilograms) and the **number of reps** you can do.
        - The app predicts the maximum weight you should be able to lift based on that information.
        - With this prediction, you can better understand your current limits and plan your training accordingly!

        **What You Can Do Here:**
        - **Single-point Prediction:** Enter your weight lifted and the number of reps to predict your bench press limit.
        - **Batch Prediction:** Upload a CSV file with multiple entries to predict the limits for multiple data points at once.

        **Why Use This App?**
        - Track your progress over time.
        - Set realistic training goals.
        - Get personalized predictions based on your own performance.
        
        Whether you're a beginner or a seasoned lifter, this app will help you optimize your training and achieve your fitness goals!
    """)

# When user selects Single-point Prediction
elif prediction_type == "Single-point Prediction":
    st.markdown("### Single-point Prediction")
    st.markdown("""
    #### Instructions 
    In this section, you can predict your **maximum bench press limit** by providing the required information in the fields below.

    The app uses this data along with a predictive model to estimate your **one-rep max (1RM)** — the maximum weight you should be able to lift for a single repetition. This prediction helps you assess your current strength level and track your progress over time.

    To get your estimated 1RM, please fill out the following fields:
    1. **Enter your personal details** (Sex, Equipment, Age, Bodyweight, Best Squat, and Best Deadlift).
    2. **Select the prediction model** you want to use for the calculation.
    
    Once you have completed all the fields, click the "Predict" button to see your estimated **one-rep max (1RM)**.
    """)

    st.markdown("#### Inputs Required For Prediction")

    # Input fields for personal data (to match the Pydantic model)
    sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
    equipment = st.selectbox("Equipment", options=["Raw", "Wraps", "Single-ply", "Multi-ply"], index=0)
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    bodyweight_kg = st.number_input("Bodyweight (kg)", min_value=0.0, value=70.0)
    best_squat_kg = st.number_input("Best Squat (kg)", min_value=0.0, value=100.0)
    best_deadlift_kg = st.number_input("Best Deadlift (kg)", min_value=0.0, value=120.0)

    # Dropdown to select the prediction model
    model_option = st.selectbox(
        "Select Prediction Model",
        options=["Random Forests", "Decision Trees", "Gradient Boosting"],
        index=0
    )

    st.markdown("#### Get Prediction")
    
    if st.button("Predict"):
        if best_deadlift_kg > 0 and best_squat_kg > 0:
            # Prepare the payload to send to FastAPI
            payload = {
                "Features": {
                    "Sex": sex,
                    "Equipment": equipment,
                    "Age": age,
                    "BodyweightKg": bodyweight_kg,
                    "BestSquatKg": best_squat_kg,
                    "BestDeadliftKg": best_deadlift_kg
                },
                "model": model_option  # Pass the selected model type
            }

            # FastAPI endpoint URL for single-point prediction
            api_url = "http://0.0.0.0:8000/predict" 

            # Make the POST request to FastAPI
            response = requests.post(api_url, json=payload)

            if response.status_code == 200:
                prediction = response.json().get("prediction", "Error in prediction")
                st.write(f"**Predicted One-Rep Max (1RM): {prediction:.2f} kg**")
            else:
                st.write("Error in prediction request.")
        else:
            st.write("Please provide valid inputs for both best_deadlift_kg and best_squat_kg.")

# When user selects Batch Prediction
elif prediction_type == "Batch Prediction":
    st.markdown("### Batch Prediction")
    st.markdown("""
    **Batch Prediction** allows you to upload a CSV file containing data for multiple individuals. 
    The app will predict the **one-rep max (1RM)** for each individual based on their provided information.

    In this section, you are able to upload data containing a number of input properties for each individual to predict the maximum bench press limit. Ensure that every property is present and well-labelled in the data before uploading.

    #### What Properties Should Be in the CSV File?
    Your CSV file should include the following columns:
    - **Sex** (Male/Female)
    - **Equipment** (Raw, Wraps, Single-ply, Multi-ply)
    - **Age** (Integer, between 18 and 100)
    - **BodyweightKg** (Float, weight in kg)
    - **BestSquatKg** (Float, best squat weight in kg)
    - **BestDeadliftKg** (Float, best deadlift weight in kg)

    #### Steps to get your predictions:
    1. **Ensure your data is well-structured**: Your CSV should include the above columns, correctly labelled.
    2. **Upload the CSV file**: Use the file uploader to upload your dataset.
    3. **Click "Predict"**: Once the data is uploaded, click the "Predict" button to get the **predicted one-rep max (1RM)** for each person in the dataset.
    """)

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file is not None:
        # Process the uploaded CSV file
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        
        # Display the first few rows of the data
        st.write("### Uploaded Data Preview")
        st.write(df.head())

        st.write("### Select a Prediction Model")
        # Dropdown to select the prediction model
        model_option = st.selectbox(
            "Select Prediction Model",
            options=["Random Forests", "Decision Trees", "Gradient Boosting"],
            index=0
        )
        # FastAPI endpoint URL for batch prediction
        api_url = "http://0.0.0.0:8000/predict_batch"  # Replace with your FastAPI URL

        # Button to trigger batch prediction
        if st.button("Predict"):
            # Prepare the payload for batch prediction (file and model)

            # Send the request to FastAPI
            response = requests.post(api_url, files={"file": uploaded_file.getvalue()}, data={"model": model_option})

            # Check if the response is successful
            if response.status_code == 200:
                predictions = response.json().get("predictions", [])
                df['predicted_limit'] = predictions
                st.write("### Prediction Results")
                st.write(df)  # Display the updated DataFrame with predictions
            else:
                st.write(f"Error in batch prediction request: {response.status_code}")




