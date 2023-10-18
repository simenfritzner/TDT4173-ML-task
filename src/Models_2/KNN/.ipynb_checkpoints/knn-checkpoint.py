from model_maker import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
      
class knn(model):
    def __init__(self,hyperparameters, X_observed, X_estimated, y, X_selected_features):
        self.model = KNeighborsRegressor(**hyperparameters)
        self.X_selected_features = X_selected_features
        self.pred_estimated = None
        self.prepare_data(X_observed, X_estimated, y, self.X_selected_features)