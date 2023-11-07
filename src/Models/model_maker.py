from utilities import *
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict, KFold

class model():
    def __init__(self):
        pass
    
    def fit(self):
        self.model.fit(self.X_train, self.y_train["pv_measurement"])
        
    def pred(self):
        max_value = self.y_train["pv_measurement"].max()
        self.prediction = self.model.predict(self.X_test)
        self.pred_estimated = self.model.predict(self.X_valid)
        self.prediction = self.prediction.clip(min = 0, max = max_value)
        
    def pred_valid(self, X_valid):
        self.pred_valid = self.model.predict(X_valid)
    
    def feature_importence_plot(self):
        self.model.feature_importances_
        plt.figure(figsize=(40,20))
        plt.barh(self.X_train.columns, self.model.feature_importances_)
    
    def corr_plot(self):
        # Calculate correlation coefficient
        corr, _ = pearsonr(self.y_valid["pv_measurement"], self.pred_estimated)
        print(f"Pearson correlation: {corr:.2f}")
        
        # Calculate the linear fit
        coefficients = np.polyfit(self.y_valid["pv_measurement"], self.pred_estimated, 1)
        polynomial = np.poly1d(coefficients)
        linear_fit = polynomial(self.y_valid["pv_measurement"])
        
        # Create scatter plot
        plt.figure(figsize=(20, 10))
        plt.scatter(self.y_valid["pv_measurement"], self.pred_estimated, label='Data points')
        plt.plot(self.y_valid["pv_measurement"], linear_fit, color='red', label=f'Linear fit: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')
        
        plt.xlabel('Actual values (y_valid)')
        plt.ylabel('Predicted values (y_pred)')
        plt.title('Correlation between Actual and Predicted Values with Linear Fit')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def cross_validate(self, X_train, y_train):
        # Cross-validation
        scores = cross_val_score(self.model, X_train, y_train["pv_measurement"], cv=5, scoring='neg_mean_absolute_error', n_jobs = -1)
        scores = -scores  # Making scores positive for easier interpretation
        self.cross_val_score_mean = scores.mean()
        self.cross_val_score = scores
        print("Cross-validation scores:", scores)
        print("Mean cross-validation score:", self.cross_val_score_mean)
        
        
    def cross_val_stack(self, X_train, y_train):
        # Setup cross-validation scheme
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        
        # Prepare to collect predictions and actual values
        predictions = []
        actuals = []
        maes = []
        # Iterate over each split
        for train_index, test_index in kf.split(X_train):
            # Split the data
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
            
            # Fit the model
            self.model.fit(X_train_fold, y_train_fold["pv_measurement"])
            
            # Predict on the test fold
            fold_predictions = self.model.predict(X_test_fold)
            
            fold_predictions = fold_predictions.clip(min = 0, max = self.y_train["pv_measurement"].max())
            
            # Calculate and store MAE for the fold
            fold_mae = mean_absolute_error(y_test_fold["pv_measurement"], fold_predictions)
            maes.append(fold_mae)
            
            # Store predictions and actual values
            predictions.append(fold_predictions)
            actuals.append(y_test_fold["pv_measurement"].values)
        
        self.cross_val_mae_mean = np.mean(maes)
        # Concatenate to have full array of cross-validated predictions and actuals
        self.cross_val_predictions = np.concatenate(predictions)
        self.cross_val_actuals = np.concatenate(actuals)
        return self.cross_val_predictions, self.cross_val_actuals, self.cross_val_mae_mean
    
    def cross_val_stack_np(self, X_train, y_train):
        # Setup cross-validation scheme
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        
        # Prepare to collect predictions and actual values
        predictions = []
        actuals = []
        maes = []
        
        # Iterate over each split
        for train_index, test_index in kf.split(X_train):
            # Split the data
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
            
            # Fit the model
            self.model.fit(X_train_fold, y_train_fold["pv_measurement"])
            
            # Predict on the test fold
            fold_predictions = self.model.predict(X_test_fold)
            
            fold_predictions = fold_predictions.clip(min = 0, max = self.y_train["pv_measurement"].max())
            
            # Calculate and store MAE for the fold
            fold_mae = mean_absolute_error(y_test_fold["pv_measurement"], fold_predictions)
            maes.append(fold_mae)
            
            # Store predictions and actual values
            predictions.append(fold_predictions)
            actuals.append(y_test_fold["pv_measurement"].values)
        
        self.cross_val_mae_mean = np.mean(maes)
        
        # Concatenate to have full array of cross-validated predictions and actuals
        self.cross_val_predictions = np.concatenate(predictions)
        self.cross_val_actuals = np.concatenate(actuals)
        
        return self.cross_val_predictions, self.cross_val_actuals, self.cross_val_mae_mean

