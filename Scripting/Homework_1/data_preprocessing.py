import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

#Load dataset
X_train = pd.read_csv("powerlifting_dataset/X_train.csv")
X_test = pd.read_csv("powerlifting_dataset/X_test.csv")
y_train = pd.read_csv("powerlifting_dataset/y_train.csv")
y_test = pd.read_csv("powerlifting_dataset/y_test.csv")




def load_data():
"""Load in the dataset"""
    data = pd.read_csv("Power_lifting.csv")
    return data


def fix_column(data, column_name):
        wrong_data = data[column_name].unique()[[*map(lambda x: type(convert_to_float(x)) != float, data[column_name].unique())]]
        corrected_data = np.array([*map(lambda x: x[:-2] + x[-1], wrong_data)]).astype('float')
        for i in range(len(wrong_data)):
            data.loc[data[column_name] == wrong_data[i], column_name] = corrected_data[i]
        data[column_name] = data[column_name].astype('float')

def pre_split_clean_data(data):
""" Fixing missing values, dropping useless columns, etc."""
    fix_column(data, 'BestSquatKg')
    return data



def convert_to_float():
    try:
        x = float(x)
    except:
        return x
    return x



def post_split_clean_data(X_train, X_test, y_train, y_test):
    mean_age = X_train['Age'].mean()
    X_train['Age'].fillna(mean_age, inplace=True)
    X_test['Age'].fillna(mean_age, inplace=True)

    input_features = pd.concat([X_train,X_test], axis=0)
    targets = pd.concat([y_train,y_test], axis=0)

    kg_features = input_features.filter(regex='Kg').columns
    input_features[kg_features] = np.abs(input_features[kg_features])

    input_features.drop(columns =["Name","playerId"],inplace=True)
    targets.drop(columns =["playerId"],inplace=True)

    return input_features, targets


def feature_engineering():
    data['RelativeSquatStrength'] = data_df['BestSquatKg'] / data_df['BodyweightKg']
    data['RelativeDeadliftStrength'] = data_df['BestDeadliftKg'] / data_df['BodyweightKg']
    return data_df

def map_equipment(data):
    """Map equipment types to their respective scores."""
    equipment_scores = {
        'Raw': 1,
        'Wraps': 2,
        'Single-ply': 3,
        'Multi-ply': 4
    }
    data['Equipment_Index'] = data['Equipment'].map(equipment_scores)
    return data



def ordinal_encoding():
    bins = [0, 18, 23, 38, 49, 59, 69, np.inf]
    labels = ['Sub-Junior', 'Junior', 'Open', 'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4']

    data_df['AgeCategory'] = pd.cut(data_df['Age'], bins=bins, labels=labels, right=False)

    ordinal_encoder = OrdinalEncoder(categories=[labels])
    data_df['AgeCategoryEncoded'] = ordinal_encoder.fit_transform(data_df[['AgeCategory']])
    return data_df
   

if __name__=="__main__":