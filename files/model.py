import pandas as import pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklern.ensemble import RandomForestClassifier

# {'max_features': 5, 'min_samples_leaf': 18, 'n_estimators': 100}

def random_search_score(X,y,Model,param_grid):
    # estimator_range = [i for i in range(10, 150, 10)]
    # max_features_range = [i for i in range(4,13)]
    # min_samples_leaf_range = [i for i in range(16,41,2)]
    # param_grid = dict(n_estimators=estimator_range, max_features=max_features_range, min_samples_leaf=min_samples_leaf_range)
    score = RandomizedSearchCV(Model, param_grid, scoring='accuracy', cv=10, n_iter=55)
    grid.fit(X,y)
    return grid.predict(X)

def create_fit_model(X,y):
    model = RandomForestClassifier(n_estimators=80, max_features=4, min_samples_leaf=10)
    model.fit(X,y)
    return model

def create_result_csv(df, results, path):
    pd.DataFrame({"PassengerId": df['PassengerId'],
                        "Survived": results}).set_index('PassengerId').to_csv(path)

def predict_result(model, X):
    result = model.predict(X)
    return result
