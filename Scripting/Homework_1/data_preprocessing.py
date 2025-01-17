import pandas as pd
import numpy as np

#Load dataset
X_train = pd.read_csv("powerlifting_dataset/X_train.csv")
X_test = pd.read_csv("powerlifting_dataset/X_test.csv")
y_train = pd.read_csv("powerlifting_dataset/y_train.csv")
y_test = pd.read_csv("powerlifting_dataset/y_test.csv")

y_test.drop(columns = ["Age",	"BodyweightKg",	"BestDeadliftKg"],inplace=True)

mean_age = X_train['Age'].mean()

#handle train data
X_train['Age'].fillna(mean_age,inplace=True)
#handle test data
X_test['Age'].fillna(mean_age,inplace=True)

input_features = pd.concat([X_train,X_test], axis=0)
targets = pd.concat([y_train,y_test], axis=0)

kg_features = input_features.filter(regex='Kg').columns
input_features[kg_features] = np.abs(input_features[kg_features])

input_features.drop(columns =["Name","playerId"],inplace=True)
targets.drop(columns =["playerId"],inplace=True)