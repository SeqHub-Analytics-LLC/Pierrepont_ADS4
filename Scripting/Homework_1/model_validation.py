import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate(train_predictions, test_predictions, y_train, y_test):
    r2_train = r2_score(train_predictions, y_train)
    r2_test = r2_score(test_predictions, y_test)
    print(f"R2:\nTrain: {r2_train}, Test: {r2_test}")

    rmse_train = np.sqrt(mean_squared_error(train_predictions, y_train))
    rmse_test = np.sqrt(mean_squared_error(test_predictions, y_test))
    print(f"RMSE:\nTrain: {rmse_train}, Test: {rmse_test}")

def plot_feature_importance(importance, names, model_name):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(f"{model_name} FEATURE IMPORTANCE")
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")