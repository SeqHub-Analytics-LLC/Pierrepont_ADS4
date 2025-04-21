# app/utils.py

import pandas as pd
import numpy as np
import joblib
from typing import Union
from pydantic import BaseModel
from models import InputFeatures


# Cleaning

def engineer_features(InputData: Union[InputFeatures, pd.DataFrame]) -> pd.DataFrame:
    # Check if InputData is a Pydantic model
    if isinstance(InputData, BaseModel):
        # Convert InputData (Pydantic model) to dictionary using .dict()
        input_dict = InputData.dict()

        # Calculate relative strength by dividing the lift weights by the body weight
        RelativeSquatStrength = input_dict['BestSquatKg'] / input_dict['BodyweightKg']
        RelativeDeadliftStrength = input_dict['BestDeadliftKg'] / input_dict['BodyweightKg']

        # Define columns and create DataFrame from the dictionary
        columns = ["Sex", "Equipment", "Age", "BodyweightKg", "BestSquatKg", "BestDeadliftKg", 
                   'RelativeSquatStrength', 'RelativeDeadliftStrength']
        df = pd.DataFrame([[input_dict['Sex'], input_dict['Equipment'], input_dict['Age'], 
                            input_dict['BodyweightKg'], input_dict['BestSquatKg'], input_dict['BestDeadliftKg'], 
                            RelativeSquatStrength, RelativeDeadliftStrength]], columns=columns)
    
    elif isinstance(InputData, pd.DataFrame):
        # Handle case where InputData is already a DataFrame
        df = InputData.copy()

        # Calculate relative strength based on the DataFrame columns
        df['RelativeSquatStrength'] = df['BestSquatKg'] / df['BodyweightKg']
        df['RelativeDeadliftStrength'] = df['BestDeadliftKg'] / df['BodyweightKg']

        # Ensure columns match the expected order
        columns = ["Sex", "Equipment", "Age", "BodyweightKg", "BestSquatKg", "BestDeadliftKg", 
                   'RelativeSquatStrength', 'RelativeDeadliftStrength']
        df = df[columns]

    else:
        raise TypeError("InputData must be either a Pydantic model or a pandas DataFrame.")

    return df
def encode_features(df, encoder_path="artifacts/ordinal_encoder.pkl"):

    # Map Age to Ordinal Categories
    bins = [0, 18, 23, 38, 49, 59, 69, np.inf]
    labels = ['Sub-Junior', 'Junior', 'Open', 'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4']
    df['AgeCategory'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # Define the mapping from sex type
    sex = {
        'Male': 1,
        'Female': 0,
    }

    df['Sex'] = df['Sex'].map(sex)

    # Load the saved encoder
    Encoder = joblib.load(encoder_path)

    #Transform the data using the loaded encoder
    cols = ['Equipment', 'AgeCategory']
    df[cols] = Encoder.transform(df[cols])
    
    #Drop redundant features
    df.drop(columns = ["Age"],inplace=True)

    return df


def scale_features(df, scaler_path="artifacts/minmax_scaler.pkl"):
    # Define columns to scale
    scale_cols = ['BodyweightKg', 'BestSquatKg', 'BestDeadliftKg','RelativeSquatStrength', 'RelativeDeadliftStrength']

    # Load the scaler and transform data
    scaler = joblib.load(scaler_path)
    df[scale_cols] = scaler.transform(df[scale_cols])
    return df
