from model_maker import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '../')
from constants import *
from utilities import *

class Random_Forest(model):
    def __init__(self, hyperparameters, X_train, X_test, y_train):
        self.X_test = X_test
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, shuffle = True, random_state = 42)
        self.model = RandomForestRegressor(**hyperparameters, random_state = 42)



def predict_y_with_random_forest(y_to_fit, y_to_predict, X_observed, X_estimated, selected_features, wanted_months, hyperparameters):
    
    # Prepare the training data
    X_train_before_prepare = prepare_X(X_observed, X_estimated, selected_features, wanted_months)
    X_train, y_train = resize_training_data(X_train_before_prepare.copy(), y_to_fit)
    
    # Prepare the prediction data
    X_y_pred = prepare_testdata_rf_a(X_train_before_prepare.copy(), selected_features, drop=True)
    X_y_pred, y_to_predict = resize_training_data(X_y_pred, y_to_predict)
    
    # Initialize and fit the Random Forest
    rf_augment_y = Random_Forest(hyperparameters, X_train, X_y_pred, y_train)
    rf_augment_y.fit()
    rf_augment_y.pred()
    
    # Ensure that the prediction length matches the y_to_predict length
    if len(rf_augment_y.prediction) != len(y_to_predict):
        raise ValueError(f"Length of predictions ({len(rf_augment_y.prediction)}) does not match length of y_to_predict ({len(y_to_predict)})")
    
    # Assign the predictions
    y_to_predict['pv_measurement'] = rf_augment_y.prediction
    
    # Combine the original and predicted data
    y_augmented = pd.concat([y_to_fit, y_to_predict], axis=0)
    
    return y_augmented
