from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=23, max_depth=7)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(random_state=34, n_estimators=500, min_samples_leaf=40, min_samples_split=90, n_jobs=-1, max_features="sqrt")
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(max_depth=3, random_state=85)
    model.fit(X_train, y_train)
    return model