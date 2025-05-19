import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def engineer_features(df):
    df['BMI'] = df['Player_Weight'] / (df['Player_Height'] / 100) ** 2
    gaps = [-float('inf'), 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
    categories = ['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III']
    df['BMI_Classification'] = pd.cut(df['BMI'], bins=gaps, labels=categories, right=False)
    df["Age_Group"] = pd.cut(
        df["Player_Age"],
        bins=[18, 22, 26, 30, 34, df["Player_Age"].max()],
        labels=["18-22", "23-26", "27-30", "31-34", "35+"],
        include_lowest=True,
    )
    return df


def encode_features(df, categorical_cols):
    encoder = OneHotEncoder(drop='first')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

    encoded_df = pd.DataFrame(encoded, columns= encoded_feature_names, index=df.index)
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def scale_features(df, numeric_cols):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df 

if __name__=="__main__":
    df=pd.read_csv("injury_data_with_categories.csv")
    categorical_cols= ["BMI_Classification", "Age_Group","Training_Surface", "Position"]
    scale_cols = ['Player_Age', 'Player_Weight', 'Player_Height', 'Previous_Injuries',
              'Training_Intensity', 'Recovery_Time']
    engf=engineer_features(df)
    print(engf.head(5))
    encf=encode_features(df, categorical_cols)
    print(encf.head(5))
    sf=scale_features(df, scale_cols)
    print(sf.head(5))





