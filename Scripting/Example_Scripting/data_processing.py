import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def load_and_clean_data(file_path):
    """Loads and cleans the dataset."""
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data.drop(columns=['date'], inplace=True)
    return data

def scale_features(X):
    """Scales features using StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir):
    """Saves preprocessed datasets."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)


