from model_maker import *
from sklearn.linear_model import LinearRegression
      
class Lin_reg(model):
    def __init__(self, X_train, X_test, y_train):
        self.model = LinearRegression()
        self.X_test = X_test
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, shuffle = True, random_state = 42)
        
