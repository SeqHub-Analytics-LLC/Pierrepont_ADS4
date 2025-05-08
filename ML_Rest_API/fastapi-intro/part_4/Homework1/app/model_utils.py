import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def engineer_features(df):
    if 'Player_Weight' in df.columns:
        df['Player_Weight'] = df['Player_Weight'].round(2)
    if 'Player_Height' in df.columns:
        df['Player_Height'] = df['Player_Height'].round(2)
    return df

def encode_features(df, categorical_cols):
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=df.index)
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def scale_features(df, numeric_cols):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

if __name__=="__main__":
    df=pd.read_csv("injury_data_with_categories.csv")
    new_df=engineer_features(df)
    print(new_df.head(5))