from model_maker import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class Random_Forest(model):
    def __init__(self, hyperparameters, X_train, X_test, y_train):
        self.X_test = X_test
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, shuffle = True, random_state = 42)
        self.model = RandomForestRegressor(**hyperparameters, random_state = 42)
        