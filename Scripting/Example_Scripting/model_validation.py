from sklearn.metrics import root_mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set."""
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2
