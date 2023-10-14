from model_maker import *
from sklearn.ensemble import RandomForestRegressor

class Random_Forest(model):
    def __init__(self,hyperparameters, X_observed, X_estimated, y, X_selected_features):
        self.model = RandomForestRegressor(**hyperparameters)
        self.X_selected_features = X_selected_features
        self.pred_estimated = None
        self.prepare_data(X_observed, X_estimated, y, self.X_selected_features)
        
    def fit(self):
        self.model.fit(self.X_train, self.y_train["pv_measurement"])
        