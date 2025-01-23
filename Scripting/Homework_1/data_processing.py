# Target - BestBenchKg
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from utils import save_encoder

# Cleaning
def convert(x):
    try:
        x = float(x)
    except:
        return x
    return x

def load_and_clean_data(file_path):
    """Loads and cleans the dataset."""
    data = pd.read_csv(file_path)
    clean_data = fix_best_squat(data)
    input_features = clean_data.drop(columns =["playerId","Name"])
    kg_features = input_features.filter(regex='Kg').columns
    input_features[kg_features] = np.abs(input_features[kg_features])
    input_features = engineer_features(input_features)
    #encoding
    full_data, new_target, oh_encoder, ordinal_encoder = encode_features(input_features)

    #save encoders
    save_encoder(oh_encoder, "model_output/oh_enocoder.pkl")
    save_encoder(ordinal_encoder, "model_output/ordinal_encoder.pkl")
    return full_data, new_target

def fix_best_squat(data):
    # fixing weird entries in BestSquatKg column
    BestSquatKg_dtype = [*map(lambda x : type(convert(x)) != float, data['BestSquatKg'].unique())]
    wrong_data_format = data['BestSquatKg'].unique()[BestSquatKg_dtype]
    correct_train_data = np.array([*map(lambda x : x[:-2] + x[-1], wrong_data_format)])
    correct_train_data = correct_train_data.astype('float')

    for i in range(len(wrong_data_format)):
        data.loc[data['BestSquatKg'] == wrong_data_format[i], 'BestSquatKg'] = correct_train_data[i]
    data['BestSquatKg'] = data['BestSquatKg'].astype('float')
    return data

def engineer_features(data):
    # Calculate relative strength by dividing the lift weights by the body weight
    data['RelativeSquatStrength'] = data['BestSquatKg'] / data['BodyweightKg']
    data['RelativeDeadliftStrength'] = data['BestDeadliftKg'] / data['BodyweightKg']

    return data

def encode_features(data):
    target = data['BestBenchKg']
    inputs = data.drop(columns = ['BestBenchKg'])
    X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

    mean_age = X_train['Age'].mean()

    #handle train data
    X_train['Age'] = X_train['Age'].fillna(mean_age)
    #handle test data
    X_test['Age'] = X_test['Age'].fillna(mean_age)

    oh_encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_train_encoded = oh_encoder.fit_transform(X_train[['Equipment','Sex']])
    X_test_encoded = oh_encoder.transform(X_test[['Equipment','Sex']])

    #GET COLUMN NAMES FOR NEWLY ENCODED VARIABLES
    encoded_columns = oh_encoder.get_feature_names_out(input_features = ['Equipment','Sex'])

    #only encoded variables
    encoded_variables = np.concat([X_train_encoded,X_test_encoded], axis=0)
    encoded_variables = pd.DataFrame(encoded_variables,columns=encoded_columns)

    #combine with other input variables
    combined_data = pd.concat([X_train,X_test], axis=0)
    full_data = pd.concat([combined_data,encoded_variables], axis=1)

    #combining targets
    new_target = pd.concat([y_train,y_test], axis=0)

    ## Bin Age
    # Define the bins for the age categories
    bins = [0, 18, 23, 38, 49, 59, 69, np.inf]
    labels = ['Sub-Junior', 'Junior', 'Open', 'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4']

    # Bin the age data
    full_data['AgeCategory'] = pd.cut(full_data['Age'], bins=bins, labels=labels, right=False)
    #Encode Bins
    ordinal_encoder = OrdinalEncoder(categories=[labels])
    full_data['AgeCategoryEncoded'] = ordinal_encoder.fit_transform(full_data[['AgeCategory']])
    
    #Drop redundant features
    full_data.drop(columns = ["Sex", "Equipment", "Age","AgeCategory"],inplace=True)

    return full_data, new_target, oh_encoder, ordinal_encoder


