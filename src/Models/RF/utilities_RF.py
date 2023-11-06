from model_maker import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class Random_Forest(model):
    def __init__(self, hyperparameters, X_train, X_test, y_train):
        self.X_test = X_test
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, shuffle = True, random_state = 42)
        self.model = RandomForestRegressor(**hyperparameters, random_state = 42)
        
        
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'loss': ['squared_error', 'absolute_error'],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.5, 0.8, 1.0],
    'min_samples_leaf': [1, 5, 10]
}

# Create a base model
gbr = GradientBoostingRegressor()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train a model with the best parameters
best_gbr = GradientBoostingRegressor(**best_params)
best_gbr.fit(X_train, y_train)

# Now you can use best_gbr to make predictions and evaluate the model
"""