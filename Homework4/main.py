import mlflow
import optuna
from preproccesing import preprocess_data
from train_model import train_and_log_model
import pickle

if __name__ == "__main__":
    

    # Preprocess the data
    preprocess_data()

    with open('pickels/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)

    with open('pickels/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)

    with open('pickels/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)

    with open('pickels/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    train_and_log_model(X_train, y_train, X_test, y_test)