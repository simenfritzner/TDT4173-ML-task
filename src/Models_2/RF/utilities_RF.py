from model_maker import *
from sklearn.ensemble import RandomForestRegressor

class Random_Forest(model):
    def __init__(self,hyperparameters, X_observed, X_estimated, y, X_selected_features):
        X_observed = X_observed.apply(lambda col: col.fillna(col.mean()), axis=0)
        X_estimated = X_estimated.apply(lambda col: col.fillna(col.mean()), axis=0)
        self.model = RandomForestRegressor(**hyperparameters, random_state = 42)
        self.X_selected_features = X_selected_features
        self.pred_estimated = None
        self.prepare_data(X_observed, X_estimated, y, self.X_selected_features)
        
    def fit(self):
        self.X_train = self.X_train.drop(columns = ['index'])
        self.X_test = self.X_test.drop(columns = ['index'])
        self.model.fit(self.X_train, self.y_train["pv_measurement"])
        