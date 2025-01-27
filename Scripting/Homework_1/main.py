from data_processing import load_data, preprocess_data
from model_training import train_decision_tree, train_random_forest, train_gradient_boosting,train_xgboost_with_optuna
from model_validation import evaluate
from joblib import dump, load



def main():
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)

    dtc_model, rf_model, gbm_model = train_decision_tree(X_train, y_train), train_random_forest(X_train, y_train), train_gradient_boosting(X_train, y_train)

    dtc_train_preds, dtc_test_preds = dtc_model.predict(X_train), dtc_model.predict(X_test)
    rf_train_preds, rf_test_preds = rf_model.predict(X_train), rf_model.predict(X_test)
    gbm_train_preds, gbm_test_preds = gbm_model.predict(X_train), gbm_model.predict(X_test)

    evaluate(dtc_train_preds, dtc_test_preds, y_train, y_test)
    evaluate(rf_train_preds, rf_test_preds, y_train, y_test)
    evaluate(gbm_train_preds, gbm_test_preds, y_train, y_test)


    final_model = train_xgboost_with_optuna(X_train, X_test, y_train, y_test)
    xgb_train_preds, xgb_test_preds = final_model.predict(X_train), final_model.predict(X_test)
    evaluate(xgb_train_preds, xgb_test_preds, y_train, y_test)

    dump(final_model, 'xgboost_model.joblib')
    

if __name__ == "__main__":
    main()