from model_maker import *
from sklearn import datasets
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sys
sys.path.insert(0, '../')
from constants import *
from utilities import *


class SVM_model(model):
    def __init__(self, X_train, X_test, y_train):
        self.X_test = X_test
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, shuffle = True, random_state = 42)
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        self.X_train = self.sc_X.fit_transform(self.X_train)
        y_train_numeric = self.y_train['pv_measurement'].values.reshape(-1, 1)
        y_train_scaled = self.sc_y.fit_transform(y_train_numeric).flatten()
        # If you need to reassign it back to the DataFrame and keep the timestamp
        self.y_train['pv_measurement'] = y_train_scaled
        print("self.y_train.shape:")
        print(self.X_train.shape)
        print("self.y_train.shape:")
        print(self.y_train.shape)
        self.X_test = self.sc_X.transform(self.X_test)
        self.model = SVR(kernel='rbf')  # RBF Kernel                                