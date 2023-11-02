from model_maker import *
from sklearn.ensemble import RandomForestRegressor

class Random_Forest(model):
    def __init__(self, hyperparameters, X_observed, X_estimated, y, X_selected_features, cross_validate = False):
        self.model = RandomForestRegressor(**hyperparameters, random_state = 42)
        self.X_selected_features = X_selected_features
        self.pred_estimated = None
        self.prepare_data(X_observed, X_estimated, y, self.X_selected_features, cross_validate)
        