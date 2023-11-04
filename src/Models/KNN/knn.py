from model_maker import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
      
class KNN_model(model):
    def __init__(self,hyperparameters, X_train, X_test, y_train):
        self.model = KNeighborsRegressor(**hyperparameters)
        self.X_test = X_test
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, shuffle = True, random_state = 42)
        
        
    def tune_hyperparameters(self, X_train, y_train, param_grid, cv=5, scoring='neg_mean_absolute_error'):
        """
        Tune hyperparameters using GridSearchCV.
        
        :param param_grid: Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
        :param cv: Determines the cross-validation splitting strategy. Default is 5.
        :param scoring: A single string to evaluate the predictions on the test set. For regression, 'neg_mean_squared_error' can be used.
        :return: Returns the best estimator and the results of the grid search.
        """
        grid_search = GridSearchCV(
            estimator=self.model, 
            param_grid=param_grid, 
            cv=cv, 
            scoring=scoring, 
            n_jobs=-1, 
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_}")
        
        return grid_search.best_estimator_, grid_search.cv_results_