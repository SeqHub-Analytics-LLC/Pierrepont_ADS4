import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

def preprocess_data():
    # Load the dataset
    df = pd.read_csv('data/breast-cancer.data', delimiter=',', names=['Class', 'age', 'menopause',
                                                                      'tumor-size', 'inv-nodes', 'node-caps',
                                                                      'deg-malig', 'breast', 'breast-quad',
                                                                      'irradiat'])

    df['node-caps'].replace('?', np.nan, inplace=True)
    df['breast-quad'].replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)

    le = LabelEncoder()
    for column in ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'breast', 'breast-quad', 'irradiat']:
        df[column] = le.fit_transform(df[column])

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open('pickels/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)

    with open('pickels/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)

    with open('pickels/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)

    with open('pickels/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

if __name__ == "__main__":
    preprocess_data()
