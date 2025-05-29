import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

def save_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def engineer_features(df):
    if 'Player_Weight' in df.columns:
        df['Player_Weight'] = df['Player_Weight'].round(2)
    if 'Player_Height' in df.columns:
        df['Player_Height'] = df['Player_Height'].round(2)
    return df

def encode_features(df, categorical_cols, path=None):
    df["Previous_Injuries"] = df["Previous_Injuries"].replace({"No": 0, "Yes": 1})
    if path:
        encoder = load_pickle(path)
    else:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoder.fit(df[categorical_cols])
        save_pickle(encoder, "artifacts/oneHotEncoder.pkl")
    encoded = encoder.transform(df[categorical_cols])
    # Create a DataFrame with the encoded features
    encoded_df = pd.DataFrame(encoded,
    columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def scale_features(df, numeric_cols, path=None):
    if path:
        scaler = load_pickle(path)
    else:
        scaler = StandardScaler()
        scaler.fit(df[numeric_cols])
        save_pickle(scaler, "artifacts/standardScaler.pkl")
    # Scale the numeric features
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

if __name__=="__main__":
    df=pd.read_csv("injury_data_with_categories.csv")
    new_df=engineer_features(df)
    encode_features(new_df, categorical_cols=["Position", "Training_Surface", ])
    scale_features(new_df, numeric_cols=["Player_Weight", "Player_Height", "Player_Age", "Training_Intensity", "Recovery_Time"])