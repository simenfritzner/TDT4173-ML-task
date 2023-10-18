from utilities import *
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

class model():
    def __init__(self):
        pass
    
    def fit(self):
        self.model.fit(self.X_train, self.y_train["pv_measurement"])
    
    def prepare_data(self, X_observed, X_estimated, y, X_selected_features):
        
        X_observed_clean = clean_df(X_observed, X_selected_features)
        X_estimated_clean = clean_df(X_estimated, X_selected_features)
        X_estimated_clean_mean = mean_df(X_estimated_clean)
        X_observed_clean_mean = mean_df(X_observed_clean)
        
        X_observed_lenght = X_observed_clean_mean.shape[0]

        X_train = pd.concat([X_observed_clean_mean, X_estimated_clean_mean], ignore_index = True)
        X_train = date_forcast_to_time(X_train)
        #X_train['almost_midnight'] = X_train['hour'].apply(assign_value)
        #X_train['sun_elevation_floored:d'] = X_train['sun_elevation:d'].apply(apply_floor_sun_elevation)
        #X_train['cloudy_day'] = ((X_train['effective_cloud_cover:p'] > 90) & (X_train['day_and_shadow'] == 1)).astype(int)
        #X_train = X_train.drop(columns = ['effective_cloud_cover:p', 'day_and_shadow'])
        # Dropping NA values generated due to lag and rolling window operations
        X_train, y = resize_training_data(X_train,y)
        self.train_test_data_split(X_train, y)
        """from sklearn.decomposition import PCA
        # Applying PCA
        pca = PCA(n_components=10)  # choose the number of components
        X_train_pca = pca.fit_transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)
        self.X_train = X_train_pca
        self.X_test = X_test_pca"""
        #self.X_train = pd.concat([self.X_train, self.X_train[X_observed_lenght:], self.X_train[X_observed_lenght:], self.X_train[X_observed_lenght:], self.X_train[X_observed_lenght:]],ignore_index = True)
        #self.y_train = pd.concat([self.y_train, self.y_train[X_observed_lenght:], self.y_train[X_observed_lenght:], self.y_train[X_observed_lenght:], self.y_train[X_observed_lenght:]], ignore_index = True)
        
        """merged_df = self.X_train.copy()
        merged_df['pv_measurement'] = y['pv_measurement'].copy()

        # Step 2: Create Lag Features
        merged_df['pv_measurement_lag1'] = merged_df['pv_measurement'].shift(1)

        # Step 3: Create Rolling Window Features (mean over a 3-hour window as an example)
        merged_df['pv_measurement_rolling_mean3'] = merged_df['pv_measurement'].rolling(window=3).mean()
        merged_df.fillna(merged_df.mean(), inplace=True)
        self.X_train = merged_df.drop(columns = ["pv_measurement"])"""
        #self.scale_data()
        
    def train_test_data_split(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.1, shuffle = True, random_state = 42)
        self.X_train = self.X_train.reset_index()
        self.X_test = self.X_test.reset_index()
        self.y_train = self.y_train.reset_index()
        self.y_test = self.y_test.reset_index()
        
    def scale_data(self):
        self.X_train = scale_df(self.X_train, True)
        self.X_test = scale_df(self.X_test, False)
        
    def pred(self, X_test = None):
        max_value = self.y_train["pv_measurement"].max()
        if X_test is None:
            # Assume self.X_test, self.y_train, and self.model are already defined and correctly set

            X_test = self.X_test.copy()  # Creating a copy to avoid modifying the original DataFrame
            """# Example usage:
            last_known_values = self.y_train['pv_measurement']  # The known 'pv_measurement' values
            predictions = generate_features_and_predict(X_test.copy(), model=self.model, last_known_values=last_known_values)


            # Predicting using the model
            self.pred_estimated = predictions"""
            self.pred_estimated = self.model.predict(X_test)

            # Clipping the predictions if necessary
            self.pred_estimated = self.pred_estimated.clip(min=0, max=max_value)  # max_value needs to be defined
            
        else:
            X_test = mean_df(X_test[self.X_selected_features]).copy()
            X_test = date_forcast_to_time(X_test).drop(columns = ["date_forecast"])
            #X_test['almost_midnight'] = X_test['hour'].apply(assign_value)
            #X_test = scale_df(X_test.drop(columns = ["date_forecast"]), False)
            self.X_test_real = X_test
            self.prediction = self.model.predict(X_test)
            self.prediction = self.prediction.clip(min = 0, max = max_value)
            
    def mae(self):
        return mean_absolute_error(self.y_test["pv_measurement"], self.pred_estimated)
    
    def corr_plot(self):
        # Calculate correlation coefficient
        corr, _ = pearsonr(self.y_test["pv_measurement"], self.pred_estimated)
        print(f"Pearson correlation: {corr:.2f}")

        # Calculate the linear fit
        coefficients = np.polyfit(self.pred_estimated, self.y_test["pv_measurement"], 1)
        polynomial = np.poly1d(coefficients)
        linear_fit = polynomial(self.pred_estimated)

        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.pred_estimated, self.y_test["pv_measurement"], label='Data points')
        plt.plot(self.pred_estimated, linear_fit, color='red', label=f'Linear fit: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')

        plt.xlabel('Predicted values (y_pred)')
        plt.ylabel('Actual values (y_test)')
        plt.title('Correlation between Actual and Predicted Values with Linear Fit')
        plt.legend()
        plt.grid(True)
        plt.show()
