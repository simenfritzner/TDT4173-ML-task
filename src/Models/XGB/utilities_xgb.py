from model_maker import *
import xgboost as xgb

class XGB_model(model):
    def __init__(self, hyperparameters, X_observed, X_estimated, y, X_selected_features):
        self.model = xgb.XGBRegressor(**hyperparameters)
        self.X_selected_features = X_selected_features
        self.pred_estimated = None
        self.prepare_data(X_observed, X_estimated, y, self.X_selected_features)