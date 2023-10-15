from model_maker import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
      
class knn(model):
    def __init__(self,hyperparameters, X_observed, X_estimated, y, X_selected_features):
        self.model = KNeighborsRegressor(**hyperparameters)
        self.X_selected_features = X_selected_features
        self.pred_estimated = None
        self.prepare_data(X_observed, X_estimated, y, self.X_selected_features)
    """def hyper_tuning(self, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_absolute_error')
        # Step 3: Fit the GridSearchCV object to find the best hyperparameters
        grid_search.fit(self.X_train, self.y_train)

        # Get the best parameters and best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        return best_model"""