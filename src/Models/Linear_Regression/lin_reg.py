from model_maker import *
from sklearn.linear_model import LinearRegression
      
class Lin_reg(model):
    def __init__(self, X_observed, X_estimated, y, X_selected_features):
        self.model = LinearRegression()
        self.X_selected_features = X_selected_features
        self.pred_estimated = None
        self.prepare_data(X_observed, X_estimated, y, self.X_selected_features)
