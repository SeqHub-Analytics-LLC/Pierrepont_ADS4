import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

def feature_engineering(df):
    df['BMI'] = df['Player_Weight'] / (df['Player_Height'] / 100) ** 2
    
    # Define BMI bins and labels
    bmi_bins = [0, 18.5, 24.9, 29.9, float('inf')]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['BMI_Classification'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_labels, right=False)
    
    # Define Age bins and labels
    age_bins = [0, 18, 25, 35, 50, float('inf')]
    age_labels = ['Teen', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']
    df['Age_Group'] = pd.cut(df['Player_Age'], bins=age_bins, labels=age_labels, right=False)
    
    return df

def encode_features(X_train, X_test, one_hot_cols):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_encoded = encoder.fit_transform(X_train[one_hot_cols])
    X_test_encoded = encoder.transform(X_test[one_hot_cols])
    with open("pickles/one_hot_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    return X_train_encoded, X_test_encoded, encoder

def scale_features(X_train, X_test, scale_cols):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[scale_cols])
    X_test_scaled = scaler.transform(X_test[scale_cols])
    with open("pickles/standard_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return X_train_scaled, X_test_scaled, scaler
