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