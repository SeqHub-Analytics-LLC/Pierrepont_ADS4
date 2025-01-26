import joblib, os
from data_preprocessing import data_preprocessing
from model_training import objective, train_model_with_optuna
from model_validation import evaluate_model
from utils import plot_param_importance, print_metrics 

if __name__ == "__main__":
    os.chdir("Scripting\Homework_1")

    data = 'data\Power_lifting.csv'

    # Perform data preprocessing
    X_train_scaled, X_test, y_train, y_test = data_preprocessing(data)

    # Train the model using Optuna
    best_model, model_path, study_path = train_model_with_optuna(X_train_scaled, y_train, n_trials=50)

    # Validate the model
    rmse, r2 = evaluate_model(best_model, X_test, y_test)
    print_metrics(rmse, r2)

    # Load the Optuna study
    study = joblib.load(study_path)

    # Plot the parameter importance
    plot_param_importance(study)

