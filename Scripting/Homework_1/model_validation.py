from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluates the model on train and test data using RMSE and R2 metrics."""
    print("Evaluating model performance...")
    
    # Predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # R2 scores
    r2_train = r2_score(y_train, train_predictions)
    r2_test = r2_score(y_test, test_predictions)

    # RMSE values
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))

    print(f"R2 Scores: Train = {r2_train:.4f}, Test = {r2_test:.4f}")
    print(f"RMSE: Train = {rmse_train:.4f}, Test = {rmse_test:.4f}")

    return {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test
    }