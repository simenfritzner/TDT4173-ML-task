import optuna
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from model_maker import *

class catboost(model):
    def __init__(self, hyperparameters, X_train, X_test, y_train):
        self.X_test = X_test
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, shuffle = True, random_state = 42)
        self.model = cb.CatBoostRegressor(**hyperparameters, random_state = 42)
        #self.model = cb.CatBoostRegressor(**hyperparameters, silent=True )
        #
        


def Hyperparametertuning_CB(X, y):
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the objective function within this scope
    def objective(trial):
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.05,  0.09397097456221813, log=True),
            "depth": trial.suggest_int("depth", 8, 16),
            "subsample": trial.suggest_float("subsample", 0.05, 0.08),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.8, 1),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 100),
        }

        model = cb.CatBoostRegressor(**params, silent=True)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        rmse = mean_squared_error(y_val, predictions, squared=False)
        return rmse

    # Create a study object and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)

    # Print and return results
    print('Best hyperparameters:', study.best_params)
    print('Best RMSE:', study.best_value)
    return study.best_params
