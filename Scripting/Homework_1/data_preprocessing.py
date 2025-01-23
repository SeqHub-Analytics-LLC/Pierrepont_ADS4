import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def data_preprocessing(paths):

    #Code scraped from the notebook
    X_train = pd.read_csv('data/' + paths[0] + '.csv')
    X_test = pd.read_csv('data/' + paths[1] + '.csv')
    y_train = pd.read_csv('data/' + paths[2] + '.csv')
    y_test = pd.read_csv('data/' + paths[3] + '.csv')

    y_test.drop(columns = ["Age",	"BodyweightKg",	"BestDeadliftKg"],inplace=True)

    def convert(x):
        try:
            x = float(x)
        except:
            return x
        return x
    
    BestSquatKg_dtype = [*map(lambda x : type(convert(x)) != float, X_train['BestSquatKg'].unique())]
    BestSquatKg_dtype2 = [*map(lambda x : type(convert(x)) != float, X_test['BestSquatKg'].unique())]

    wrong_train_data = X_train['BestSquatKg'].unique()[BestSquatKg_dtype]
    wrong_test_data = X_test['BestSquatKg'].unique()[BestSquatKg_dtype2]

    correct_train_data = np.array([*map(lambda x : x[:-2] + x[-1], wrong_train_data)])
    correct_train_data = correct_train_data.astype('float')
    for i in range(len(wrong_train_data)):
        X_train.loc[X_train['BestSquatKg'] == wrong_train_data[i], 'BestSquatKg'] = correct_train_data[i]

    X_train['BestSquatKg'] = X_train['BestSquatKg'].astype('float')

    mean_age = X_train['Age'].mean()
    X_train['Age'].fillna(mean_age,inplace=True)
    X_test['Age'].fillna(mean_age,inplace=True)

    input_features = pd.concat([X_train,X_test], axis=0)
    targets = pd.concat([y_train,y_test], axis=0)

    kg_features = input_features.filter(regex='Kg').columns
    input_features[kg_features] = np.abs(input_features[kg_features])

    input_features.drop(columns =["Name","playerId"],inplace=True)
    targets.drop(columns =["playerId"],inplace=True)

    train_ = input_features.iloc[:X_train.shape[0],:]
    target_ = targets.iloc[:X_train.shape[0]]

    age_bins = [0, 18, 23, 38, 49, 59, 69, float('inf')]
    age_labels = ['18 and under', '19-23', '24-38', '39-49', '50-59', '60-69', '70+']

    train_['AgeGroup'] = pd.cut(train_['Age'], bins=age_bins, labels=age_labels, right=False)

    data_df = pd.concat([input_features,targets],axis=1)
    
    data_df['RelativeSquatStrength'] = data_df['BestSquatKg'] / data_df['BodyweightKg']
    data_df['RelativeDeadliftStrength'] = data_df['BestDeadliftKg'] / data_df['BodyweightKg']

    equipment_scores = {
    'Raw': 1,
    'Wraps': 2,
    'Single-ply': 3,
    'Multi-ply': 4
    }

    data_df['Equipment_Index'] = data_df['Equipment'].map(equipment_scores)

    data_df["Equipment"].unique(), data_df['Sex'].unique()

    data_df = pd.get_dummies(data_df, columns=['Equipment',"Sex"], drop_first=True)

    bins = [0, 18, 23, 38, 49, 59, 69, np.inf]
    labels = ['Sub-Junior', 'Junior', 'Open', 'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4']

    data_df['AgeCategory'] = pd.cut(data_df['Age'], bins=bins, labels=labels, right=False)

    target = data_df['BestBenchKg']
    input_features = data_df.drop(columns =["BestBenchKg","Age","AgeCategory"])

    def date_processing(data):
        data['date'] = pd.to_datetime(data['date'])
        data['day'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        data.drop(columns=['date'], inplace=True)
        return data

    data_df.to_csv('processed_data/data.csv', index=False)
    
    y_train, y_test = target.iloc[:X_train.shape[0]],target.iloc[-5000:]

    X_train, X_test =  input_features.iloc[:X_train.shape[0],:], input_features.iloc[-5000:,:]

    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    X_test = pd.DataFrame(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    
    X_train_scaled.to_csv('processed_data/X_train.csv', index=False)
    X_test.to_csv('processed_data/X_test.csv', index=False)
    y_train.to_csv('processed_data/y_train.csv', index=False)
    y_test.to_csv('processed_data/y_test.csv', index=False)

    return data_df, X_train_scaled, X_test, y_train, y_test

    