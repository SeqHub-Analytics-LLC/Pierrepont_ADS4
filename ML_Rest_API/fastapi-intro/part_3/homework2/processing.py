import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

def load_and_clean_data(file_path):
    """Loads and cleans the dataset."""
    df = pd.read_csv(file_path)
    df['Player_Weight'] = df['Player_Weight'].round(2)
    df['Player_Height'] = df['Player_Height'].round(2)
    df['Training_Intensity'] = df['Training_Intensity'].round(2)
    return df

def create_features(df):
    # Calculate the Body Mass Index (BMI)
    df['BMI'] = df['Player_Weight'] / (df['Player_Height'] / 100) ** 2
    # Defining gaps for BMI classification
    gaps = [-float('inf'), 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
    categories = ['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III']
    # Create "BMI_Classification" column 
    df['BMI_Classification'] = pd.cut(df['BMI'], bins=gaps, labels=categories, right=False)
    df["Age_Group"] = pd.cut(df["Player_Age"], bins=[18, 22, 26, 30, 34, 120],
        labels=["18-22", "23-26", "27-30", "31-34", "35+"],
        include_lowest=True,
    )
    return df

def encode_features(df, use_saved=False, encoder_path="data/artifacts/one_hot_encoder.pkl"):
    """
    Encodes categorical features using OneHotEncoder.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data to be encoded.
        use_saved (bool): If True, use a saved encoder at encoder_path; if False, fit a new encoder.
        encoder_path (str): File path to load/save the encoder (default: "encoder.pkl").
    """
    # Step 2: Define categorical columns to encode
    one_hot_cols = ["BMI_Classification", "Age_Group", "Training_Surface", "Position"]

    # Step 3 & 4: Load a saved encoder or fit a new one based on the use_saved flag
    if use_saved:
        # Load the saved encoder from the specified file path
        encoder = joblib.load(encoder_path)
        # Transform the data using the loaded encoder
        df_encoded = encoder.transform(df[one_hot_cols])
    else:
        # Initialize the encoder and fit it on the data
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        df_encoded = encoder.fit_transform(df[one_hot_cols])
        # Optionally save the encoder for future use
        joblib.dump(encoder, encoder_path)
    
    # Step 5: Get the encoded feature names
    encoded_feature_names = encoder.get_feature_names_out(one_hot_cols)
    
    # Step 6: Convert the encoded array to a DataFrame
    encoded_df = pd.DataFrame(df_encoded, columns=encoded_feature_names, index=df.index)
    
    # Step 7: Drop original categorical columns
    df_clean = df.drop(columns=one_hot_cols)
    
    # Step 8: Concatenate the encoded columns with the rest of the DataFrame
    df_final = pd.concat([df_clean, encoded_df], axis=1)

    return df_final, encoder


def scale_features(X_train, X_test, use_saved=False, scaler_path="data/artifacts/standard_scaler.pkl"):
    """
    Scales features using a StandardScaler.

    Parameters:
        X_train (pd.DataFrame): Training dataset.
        X_test (pd.DataFrame): Testing dataset.
        use_saved (bool): If True, load a saved scaler from scaler_path; 
                          otherwise, fit a new scaler on X_train.
        scaler_path (str): File path to load/save the scaler (default: "scaler.pkl").
    """
    # Define columns to scale
    scale_cols = ['Player_Age', 'Player_Weight', 'Player_Height', 
                  'Previous_Injuries', 'Training_Intensity', 'Recovery_Time']
    
    if use_saved:
        # Load the scaler from the specified file and transform both train and test data
        scaler = joblib.load(scaler_path)
        X_scaled2 = scaler.transform(X_test[scale_cols])
        return X_scaled2
    else:
        # Initialize and fit a new scaler on the training data
        scaler = StandardScaler()
        X_scaled1 = scaler.fit_transform(X_train[scale_cols])
        X_scaled2 = scaler.transform(X_test[scale_cols])
        # Save the fitted scaler for future use
        joblib.dump(scaler, scaler_path)
        return X_scaled1, X_scaled2, scaler


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir='data/preprocessed_data'):
    """Saves preprocessed datasets."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directory if it doesn't exist
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)


