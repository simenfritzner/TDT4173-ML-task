from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np


def agumenting_time(df):
    df["new_time"] = pd.to_datetime(df['date_forecast'])
    df['hour'] = df['new_time'].dt.hour
    df['minute'] = df['new_time'].dt.minute
    df["day"]  = df['new_time'].dt.day
    df["month"]  = df['new_time'].dt.month
    df['time_decimal'] = df['hour'] + df['minute'] / 60.0
    phase_adjustment = (np.pi/2) - 11 * (2 * np.pi / 24)
    df['hour_sin'] = np.sin(df['time_decimal'] * (2. * np.pi / 24) + phase_adjustment)
    df['hour_cos'] = np.cos(df['time_decimal'] * (2. * np.pi / 24) + phase_adjustment)
    return df

def direct_rad_div_diffuse_rad(df):
    df['dif_dat_rad'] = 0.0
    condition = df['diffuse_rad:W'] != 0
    df.loc[condition, 'dif_dat_rad'] = df.loc[condition, 'direct_rad:W'] / df.loc[condition, 'diffuse_rad:W']
    return df

def get_hyperparameters_for_rf(x_observed, x_estimated, y, selected_features ):
    X_train = pd.concat([clean_df(x_observed, selected_features), clean_df(x_estimated, selected_features)])
    X_train, y_train = resize_training_data(X_train,y)
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create a Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)

    # Create the scorer
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Create the grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scorer, cv=5)

    # Fit the grid search
    grid_search.fit(X_train, y_train["pv_measurement"])

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters b: {best_params}")
    return best_params