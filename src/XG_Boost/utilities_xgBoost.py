import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

class xgBoost():
    def __init__(self, X_observed, X_estimated, y, X_selected_features):
        self.model = XGBRegressor()
        self.X_selected_features = X_selected_features
        self.pred_estimated = None
        self.prepare_data(X_observed, X_estimated, y, self.X_selected_features)
        
    def fit(self):
        self.model.fit(self.X_train, self.y_train["pv_measurement"])
    
    def prepare_data(self, X_observed, X_estimated, y, X_selected_features):
        
        X_observed_clean = clean_df(X_observed, X_selected_features)
        X_estimated_clean = clean_df(X_estimated, X_selected_features)
        X_estimated_clean_mean = mean_df(X_estimated_clean)
        X_observed_clean_mean = mean_df(X_observed_clean)
        
        X_train = pd.concat([X_observed_clean_mean, X_estimated_clean_mean])
        X_train, y = resize_training_data(X_train,y)
        self.train_test_data_split(X_train, y)
        #self.scale_data()
        
    def train_test_data_split(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.05, shuffle = False)
        
    def scale_data(self):
        self.X_train = scale_df(self.X_train)
        self.X_valid = scale_df(self.X_valid)
        self.X_test = scale_df(self.X_test)
        
    def pred(self, X_test = None):
        max_value = self.y_train["pv_measurement"].max()
        if X_test is None:
            X_test = self.X_test
            self.pred_estimated = self.model.predict(X_test)
            self.pred_estimated = self.pred_estimated.clip(min = 0, max = max_value)
            
        else:
            X_test = mean_df(X_test[self.X_selected_features]).drop(columns = ["date_forecast"]).copy()
            X_test = scale_df(X_test)
            print(X_test)
            self.pred = self.model.predict(X_test)
            self.pred = self.pred.clip(min = 0, max = max_value)
            
    def mae(self):
        if self.pred_estimated is None:
            self.pred()
        return mean_absolute_error(self.y_test["pv_measurement"], self.pred_estimated)
    
            
#Scales all the feature value in a way they take a simmilar range
def scale_df(df):
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df

#Removes all features from a df except selected_features
def clean_df(df, selected_features):
    return df[selected_features]

#Function which resizes the training data such that only the rows with the same date and time for weather is kept.
#X_train is either observed or forcasted weather and y_train is how much energy is produced. 
#y_features are a list containing the column names of y_train
#X_date_feature is the feature name which the date and time for the weather is savew. This will probably always be "date_forecast" and may be changed
def resize_training_data(X_train, y_train):
    y_features = y_train.columns.tolist()
    X_date_feature = "date_forecast"
    
    merged = pd.merge(X_train, y_train,left_on=X_date_feature, right_on='time', how='inner')
    y_train_resized = merged[y_features]
    columns_to_drop = y_features + [X_date_feature]
    X_train_resized = merged.drop(columns = columns_to_drop)
    return X_train_resized, y_train_resized

#Saves the predictions in proper format, y_pred needs to contain predicitions for all 3 locatoins
def submission(filename, y_pred):
    test = pd.read_csv('CSV/test.csv')
    submission = pd.read_csv('CSV/sample_submission.csv')
    test['prediction'] = y_pred
    submission = submission[['id']].merge(test[['id', 'prediction']], on='id', how='left')
    submission.to_csv(filename, index=False)
    
#Splits the training data such that it is training set is observed and some estimated, valid is some estimated and test is some estimated
def training_data_split(X_observed_clean, X_estimated_clean_mean):
    X_train_estimated = X_estimated_clean_mean[:int(X_estimated_clean_mean.shape[0] * 3 / 4)]
    X_valid = X_estimated_clean_mean[int(X_estimated_clean_mean.shape[0] * 3 / 4):int(X_estimated_clean_mean.shape[0] * 9 / 10)]
    X_test = X_estimated_clean_mean[int(X_estimated_clean_mean.shape[0] * 9 / 10):]
    
    X_train = pd.concat([X_observed_clean, X_train_estimated])
    return X_train, X_valid, X_test
    
#A function which takes the mean out of every 4th column and saves it on the time on the time of the 4th. Makes it so it is every hour.
#TODO: Should be swapped for Gustavs code!
def mean_df(df):
    # Assuming df is your DataFrame and 'date_forecast' is your date column
    # Making a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()
    
    # Step 1: Keeping every 4th row in the date column
    date_column = df_copy['date_forecast'].iloc[::4]
    
    # Step 2: Creating a grouping key
    grouping_key = np.floor(np.arange(len(df_copy)) / 4)
    
    # Step 3: Group by the key and calculate the mean, excluding the date column
    averaged_data = df_copy.drop(columns=['date_forecast']).groupby(grouping_key).mean()
    
    # Step 4: Reset index and merge the date column
    averaged_data.reset_index(drop=True, inplace=True)
    averaged_data['date_forecast'] = date_column.values
    return averaged_data


        