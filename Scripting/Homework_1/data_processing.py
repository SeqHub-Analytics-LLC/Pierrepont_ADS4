import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_split_data():
    data = pd.read_csv("powerlifting.csv")
    
    X = data.drop(columns=["BestBenchKg", "playerId", "Name"])  # Drop target and non-feature columns
    y = data["BestBenchKg"]  # Set target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test, y_train, y_test):
    mean_age = X_train['Age'].mean()

    X_train['Age'].fillna(mean_age, inplace=True)
    X_test['Age'].fillna(mean_age, inplace=True)

    X_train.drop_duplicates(inplace=True)
    X_test.drop_duplicates(inplace=True)

    X_train['BestSquatKg'] = X_train['BestSquatKg'].astype('float')

    return X_train, X_test, y_train, y_test