from utilities import *
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

class model():
    def __init__(self):
        pass
    
    def fit(self):
        self.model.fit(self.X_train, self.y_train["pv_measurement"])
        
    def pred(self):
        #max_value = self.y_train["pv_measurement"].max()
        self.prediction = self.model.predict(self.X_test)
        self.pred_estimated = self.model.predict(self.X_valid)
        #self.prediction = self.prediction.clip(min = 0, max = max_value)
    
    def feature_importence_plot(self):
        self.model.feature_importances_
        plt.figure(figsize=(20,10))
        plt.barh(self.X_train.columns, self.model.feature_importances_)
    
    def corr_plot(self):
        # Calculate correlation coefficient
        corr, _ = pearsonr(self.y_test["pv_measurement"], self.pred_estimated)
        print(f"Pearson correlation: {corr:.2f}")
        
        # Calculate the linear fit
        coefficients = np.polyfit(self.y_test["pv_measurement"], self.pred_estimated, 1)
        polynomial = np.poly1d(coefficients)
        linear_fit = polynomial(self.y_test["pv_measurement"])
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test["pv_measurement"], self.pred_estimated, label='Data points')
        plt.plot(self.y_test["pv_measurement"], linear_fit, color='red', label=f'Linear fit: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')
        
        plt.xlabel('Actual values (y_test)')
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

