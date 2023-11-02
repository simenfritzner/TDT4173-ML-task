from sklearn.base import clone
import numpy as np
import pandas as pd

class Stacker(model):
    def __init__(self, base_models, meta_model, X_observed, X_estimated, y, X_selected_features, cross_validate=False):
        self.base_models = base_models
        self.meta_model = clone(meta_model)
        self.X_selected_features = X_selected_features
        self.prepare_data(X_observed, X_estimated, y, self.X_selected_features, cross_validate)
        
    def fit(self):
        # Train base models
        self.base_models_ = [list() for x in self.base_models]
        
        for i, base_model in enumerate(self.base_models):
            instance = clone(base_model)
            instance.fit(self.X_train, self.y_train)
            self.base_models_[i].append(instance)
            
        # Generate meta-features for training set
        meta_features = self.generate_meta_features(self.X_train)
        
        # Fit the meta-model
        self.meta_model.fit(meta_features, self.y_train["pv_measurement"])
        
    def generate_meta_features(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return meta_features
    
    def pred(self, X_test=None):
        # Generate meta-features for test set
        meta_features = self.generate_meta_features(X_test)
        
        # Predict using meta-model
        self.prediction = self.meta_model.predict(meta_features)
