from model_maker import *
import xgboost as xgb

class XGB_model(model):
    def __init__(self, hyperparameters, X_train, X_test, y_train):
        self.X_test = X_test
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, shuffle = True, random_state = 42)
        self.model = xgb.XGBRegressor(**hyperparameters)