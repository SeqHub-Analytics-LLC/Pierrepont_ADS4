from google.colab import drive
drive.mount('/content/drive')


import os
os.chdir("drive/MyDrive/Pierrepont 2024/Intro to ML/datasets")

import pandas as pd
import numpy as np

#Load dataset
X_train = pd.read_csv("powerlifting_dataset/X_train.csv")
X_test = pd.read_csv("powerlifting_dataset/X_test.csv")
y_train = pd.read_csv("powerlifting_dataset/y_train.csv")
y_test = pd.read_csv("powerlifting_dataset/y_test.csv")

#Preview the dataset
X_train.head(4)

y_train.head(4)

X_test.head(4)

y_test.head(4)

y_test.drop(columns = ["Age",	"BodyweightKg",	"BestDeadliftKg"],inplace=True)
y_test.head(4)

print(np.sum(X_test["playerId"].values == y_test['playerId']) == X_test.shape[0])
print(np.sum(X_train["playerId"].values == y_train['playerId']) == X_train.shape[0])

X_train.info()

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

wrong_train_data, wrong_test_data

correct_train_data = np.array([*map(lambda x : x[:-2] + x[-1], wrong_train_data)])
correct_train_data = correct_train_data.astype('float')
for i in range(len(wrong_train_data)):
    X_train.loc[X_train['BestSquatKg'] == wrong_train_data[i], 'BestSquatKg'] = correct_train_data[i]

X_train['BestSquatKg'] = X_train['BestSquatKg'].astype('float')
X_train['BestSquatKg'].dtype

X_train['Age'].describe()

# Plot histogram
import seaborn as sns
sns.histplot(data = X_train['Age'], kde = True)

mean_age = X_train['Age'].mean()

#handle train data
X_train['Age'].fillna(mean_age,inplace=True)
#handle test data
X_test['Age'].fillna(mean_age,inplace=True)

print(X_train.duplicated().sum())
print(X_test.duplicated().sum())

input_features = pd.concat([X_train,X_test], axis=0)
targets = pd.concat([y_train,y_test], axis=0)

input_features.describe()

kg_features = input_features.filter(regex='Kg').columns
input_features[kg_features] = np.abs(input_features[kg_features])

input_features.drop(columns =["Name","playerId"],inplace=True)
targets.drop(columns =["playerId"],inplace=True)

train_ = input_features.iloc[:X_train.shape[0],:]
target_ = targets.iloc[:X_train.shape[0]]

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn
sns.set_style("whitegrid")

# Plot the distribution of BestBenchKg
plt.figure(figsize=(10, 6))
sns.histplot(target_['BestBenchKg'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Best Bench Press (Kg)')
plt.xlabel('Best Bench Press (Kg)')
plt.ylabel('Frequency')
plt.show()


## Your implementation here

# Plot the distribution of BestBenchKg between male and female lifters
plt.figure(figsize=(10, 6))
sns.boxplot(x=train_['Sex'], y=target_['BestBenchKg'], palette='coolwarm')
plt.title('Comparison of Best Bench Press (Kg) Between Male and Female Lifters')
plt.xlabel('Sex')
plt.ylabel('Best Bench Press (Kg)')
plt.show()


#Your implementation here

#solution
plt.figure(figsize=(12, 8))
sns.regplot(x=train_['BodyweightKg'], y=target_['BestBenchKg'], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Relationship between Bodyweight and Best Bench Press')
plt.xlabel('Bodyweight (Kg)')
plt.ylabel('Best Bench Press (Kg)')
plt.show()


print(f"Correlation Matrix between Bodyweight and Best Bench Press\n {np.corrcoef(train_['BodyweightKg'],target_['BestBenchKg'])}")

## Your Implementation Here



#solution
plt.figure(figsize=(10, 6))
sns.barplot(x=train_['Equipment'], y=target_['BestBenchKg'], ci=None, palette='Set2')
#sns.boxplot(x=train_['Equipment'], y=target_['BestBenchKg'], palette='Set2')
plt.title('Impact of Equipment on Best Bench Press (Kg)')
plt.xlabel('Equipment')
plt.ylabel('Average Best Bench Press (Kg)')
plt.show()


## Your implementation here

# Define age bins and labels
age_bins = [0, 18, 23, 38, 49, 59, 69, float('inf')]
age_labels = ['18 and under', '19-23', '24-38', '39-49', '50-59', '60-69', '70+']

# Categorize the data into age groups
train_['AgeGroup'] = pd.cut(train_['Age'], bins=age_bins, labels=age_labels, right=False)

# Plot the average BestBenchKg across age groups
plt.figure(figsize=(12, 8))
sns.barplot(x=train_['AgeGroup'], y= target_['BestBenchKg'], ci=None, palette='viridis')
plt.title('Best Bench Press (Kg) Across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Average Best Bench Press (Kg)')
plt.xticks(rotation=45)
plt.show()


data_df = pd.concat([input_features,targets],axis=1)
data_df.head()

# Calculate relative strength by dividing the lift weights by the body weight
data_df['RelativeSquatStrength'] = data_df['BestSquatKg'] / data_df['BodyweightKg']
data_df['RelativeDeadliftStrength'] = data_df['BestDeadliftKg'] / data_df['BodyweightKg']

# Define the mapping from equipment type to index
equipment_scores = {
    'Raw': 1,
    'Wraps': 2,
    'Single-ply': 3,
    'Multi-ply': 4
}

# Map the equipment types to their respective scores
data_df['Equipment_Index'] = data_df['Equipment'].map(equipment_scores)

#Check unique categories in categorical columns
data_df["Equipment"].unique(), data_df['Sex'].unique()

data_df = pd.get_dummies(data_df, columns=['Equipment',"Sex"], drop_first=True)
data_df.head()

# Define the bins for the age categories
bins = [0, 18, 23, 38, 49, 59, 69, np.inf]
labels = ['Sub-Junior', 'Junior', 'Open', 'Masters 1', 'Masters 2', 'Masters 3', 'Masters 4']

# Bin the age data
data_df['AgeCategory'] = pd.cut(data_df['Age'], bins=bins, labels=labels, right=False)
data_df['AgeCategory']

# Now we will use OrdinalEncoder to encode these categories
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(categories=[labels])
data_df['AgeCategoryEncoded'] = ordinal_encoder.fit_transform(data_df[['AgeCategory']])
data_df['AgeCategoryEncoded']

data_df.head(3)

target = data_df['BestBenchKg']
input_features = data_df.drop(columns =["BestBenchKg","Age","AgeCategory"])

#Targets
y_train, y_test = target.iloc[:X_train.shape[0]],target.iloc[-5000:]
#Input features
X_train, X_test =  input_features.iloc[:X_train.shape[0],:], input_features.iloc[-5000:,:]

print(X_train.shape, X_test.shape), print(y_train.shape, y_test.shape)

from sklearn.metrics import mean_squared_error, r2_score

def evaluate(train_predictions, test_predictions):

    r2_train = r2_score(train_predictions, y_train)
    r2_test = r2_score(test_predictions, y_test)
    print(f"R2:\nTrain: {r2_train}, Test: {r2_test}")

    rmse_train = np.sqrt(mean_squared_error(train_predictions, y_train))
    rmse_test = np.sqrt(mean_squared_error(test_predictions, y_test))
    print(f"RMSE:\nTrain: {rmse_train}, Test: {rmse_test}")

from matplotlib import pyplot as plt
def plot_feature_importance(importance, names, model_name):
    #create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # create a dataframe using a dictionary
    data = {'feature_names':feature_names, 'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'],ascending = False, inplace=True)

    #define size of bar plot
    plt.figure(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'], y = fi_df['feature_names'])
    plt.title(model_name + "FEATURE IMPORTANCE")

    #Add chart labels
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")

from sklearn.tree import  DecisionTreeRegressor

dtc_model =  DecisionTreeRegressor(random_state=23, max_depth=7)

dtc_model.fit(X_train, y_train)

dtc_train_predictions = dtc_model.predict(X_train)
dtc_test_predictions = dtc_model.predict(X_test)

evaluate(dtc_train_predictions, dtc_test_predictions)

importances = dtc_model.feature_importances_

# Print the importance of each feature
for feature, importance in zip(X_train.columns, importances):
    print(f"{feature}: {importance:.4f}")

plot_feature_importance(importances , names = X_train.columns, model_name= "Decision Trees")

#To-Do: Hyper-parameter Optimization


from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor(random_state=34,n_estimators=500, min_samples_leaf=40,
                                 min_samples_split=90, n_jobs=-1, max_features="sqrt")

RF_model.fit(X_train, y_train)

RF_train_predictions = RF_model.predict(X_train)
RF_test_predictions = RF_model.predict(X_test)

evaluate(RF_train_predictions, RF_test_predictions)

importances = RF_model.feature_importances_

# Print the importance of each feature
for feature, importance in zip(X_train.columns, importances):
    print(f"{feature}: {importance:.4f}")

plot_feature_importance(importances , names = X_train.columns, model_name= "Random Forests ")

#To-Do: Hyper-parameter Optimization


from IPython.display import YouTubeVideo

YouTubeVideo('zTXVo2Kmi8Q', width=800, height=500)

from sklearn.ensemble import GradientBoostingRegressor

GBM_model = GradientBoostingRegressor(max_depth=3,random_state=85)

GBM_model.fit(X_train, y_train)

GBM_train_predictions = GBM_model.predict(X_train)
GBM_test_predictions = GBM_model.predict(X_test)

evaluate(GBM_train_predictions, GBM_test_predictions)

importances = GBM_model.feature_importances_

# Print the importance of each feature
for feature, importance in zip(X_train.columns, importances):
    print(f"{feature}: {importance:.4f}")

plot_feature_importance(importances , names = X_train.columns, model_name= "Gradient Boosting Trees")

#To-Do: Hyper-parameter Optimization


