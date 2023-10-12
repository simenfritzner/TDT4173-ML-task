import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

#Function which resizes the training data such that only the rows with the same date and time for weather is kept.
#X_train is either observed or forcasted weather and y_train is how much energy is produced. 
#y_features are a list containing the column names of y_train
#X_date_feature is the feature name which the date and time for the weather is savew. This will probably always be "date_forecast" and may be changed
def resize_trainingdata(X_train, y_train, X_date_feature, y_features):
    merged = pd.merge(X_train, y_train,left_on=X_date_feature, right_on='time', how='inner')
    y_train_resized = merged[y_features]
    columns_to_drop = y_features + [X_date_feature]
    X_train_resized = merged.drop(columns = columns_to_drop)
    return X_train_resized, y_train_resized


def submission(filename, y_pred):
    test = pd.read_csv('CSV/test.csv')
    submission = pd.read_csv('CSV/sample_submission.csv')
    test['prediction'] = y_pred
    submission = submission[['id']].merge(test[['id', 'prediction']], on='id', how='left')
    submission.to_csv(filename, index=False)
    
def pred(X_observed, X_estimated, y, selected_features, X_test = None):
    y_features = ["time", "pv_measurement"]
    
    X_observed_clean = mean_df(X_observed[selected_features])
    X_estimated_clean = mean_df(X_estimated[selected_features])
    #Training, validation and test
    X_train_estimated = X_estimated_clean[:int(X_estimated_clean.shape[0] * 3 / 4)]
    #X_valid_estimated = X_estimated_clean[int(X_estimated_clean.shape[0] * 3 / 4):int(X_estimated_clean.shape[0] * 9 / 10)]
    X_test_estimated = X_estimated_clean[int(X_estimated_clean.shape[0] * 9 / 10):]
    
    #Training a Linear regression model on X_observed_a and testing it on X_estimated_a and evaluating it on MAE, PURELY for testing!
    #See below for how its done when submitting
    X_train_observed_resized, y_train_observed = resize_trainingdata(X_observed_clean, y, "date_forecast", y_features)
    X_train_estimated_resized, y_train_estimated = resize_trainingdata(X_train_estimated, y, "date_forecast", y_features)
    #X_valid_estimated_a_resized, y_valid_estimated_a = resize_trainingdata(X_valid_estimated_a, train_a, "date_forecast", y_features)
    X_test_estimated_resized, y_test_estimated = resize_trainingdata(X_test_estimated, y, "date_forecast", y_features)
    
    X_train = pd.concat([X_train_observed_resized, X_train_estimated_resized ], ignore_index=True)
    y_train = pd.concat([y_train_observed, y_train_estimated], ignore_index = True)
    
    #Scaling the data for more fair comparions and faster convergence, ChatGPT
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is None:
        X_test_scaled = scaler.fit_transform(X_test_estimated_resized)
    else:
        X_test = mean_df(X_test[selected_features]).drop(columns = ["date_forecast"]).copy()
        X_test_scaled = scaler.fit_transform(X_test)
    
    #Training the model
    reg = LinearRegression()
    reg.fit(X_train_scaled, y_train["pv_measurement"])
    
    # Make predictions
    y_pred = reg.predict(X_test_scaled)
    return y_pred, y_test_estimated

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