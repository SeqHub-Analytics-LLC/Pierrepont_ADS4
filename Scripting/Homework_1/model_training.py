from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def evaluate(train_predictions, test_predictions, y_train, y_test):
    r2_train = r2_score(y_train, train_predictions)
    r2_test = r2_score(y_test, test_predictions)
    print(f"R2:\nTrain: {r2_train}, Test: {r2_test}")
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    print(f"RMSE:\nTrain: {rmse_train}, Test: {rmse_test}")

def plot_feature_importance(importance, names, model_name):
    # Create a dataframe for feature importance and names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the dataframe by feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Plot the feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(model_name + " FEATURE IMPORTANCE")

    # Add chart labels
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")
    plt.tight_layout()
    plt.show()