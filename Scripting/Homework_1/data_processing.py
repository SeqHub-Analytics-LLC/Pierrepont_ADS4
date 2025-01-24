import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import encode_age_category

def convert(x):
    try:
        x = float(x)
    except:
        return x
    return x

def load_data():

    data = pd.read_csv("powerlifting.csv")
    
    X = data.drop(columns=["BestBenchKg", "playerId", "Name"])
    y = data["BestBenchKg"]  # Set target variable
    
  #  X = encode_age_category(X)

    BestSquatKg_dtype = [*map(lambda x : type(convert(x)) != float, X['BestSquatKg'].unique())]
    wrong_train_data = X['BestSquatKg'].unique()[BestSquatKg_dtype]
    correct_data = np.array([*map(lambda x : x[:-2] + x[-1], wrong_train_data)])
    correct_data= correct_data.astype('float')

    for i in range(len(wrong_train_data)):
        X.loc[X['BestSquatKg'] == wrong_train_data[i], 'BestSquatKg'] = correct_data[i]
    
    X['BestSquatKg'] = X['BestSquatKg'].astype('float')


    data.dropna(subset=['BestBenchKg'], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test, y_train, y_test):
    mean_age = X_train['Age'].mean()
    X_train['Age'].fillna(mean_age, inplace=True)
    X_test['Age'].fillna(mean_age, inplace=True)

    categorical_columns = ['Sex', 'Equipment']  
    X_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)
    
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    X_train['BestSquatKg'] = X_train['BestSquatKg'].astype('float')
    X_test['BestSquatKg'] = X_test['BestSquatKg'].astype('float')
    
    X_train.drop_duplicates(inplace=True)
    X_test.drop_duplicates(inplace=True)
    
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train,_,_,_ = load_data()
    print(X_train.head())
    