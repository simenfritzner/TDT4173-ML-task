from model_maker import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
      
class KNN_model(model):
    def __init__(self,hyperparameters, X_train, X_test, y_train):
        self.model = KNeighborsRegressor(**hyperparameters)
        self.X_test = X_test
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, shuffle = True, random_state = 42)