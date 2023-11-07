import pandas as pd
from utilities import *
#Loading the y-paramter from file, might be changed to y_a etc.
y_a = pd.read_parquet('../../Data/Data_and_task/A/train_targets.parquet')
y_b = pd.read_parquet('../../Data/Data_and_task/B/train_targets.parquet')
y_c = pd.read_parquet('../../Data/Data_and_task/C/train_targets.parquet')
y_b = y_b.dropna()
y_c=y_c.dropna()
y_c, _ = augment_y_c(y_c)

#Loading estimated/forecasted training_weather from file
X_estimated_a = pd.read_parquet('../../Data/Data_and_task/A/X_train_estimated.parquet')
X_estimated_b = pd.read_parquet('../../Data/Data_and_task/B/X_train_estimated.parquet')
X_estimated_c = pd.read_parquet('../../Data/Data_and_task/C/X_train_estimated.parquet')

#Loading observed weather from file
X_observed_a = pd.read_parquet('../../Data/Data_and_task/A/X_train_observed.parquet')
X_observed_b = pd.read_parquet('../../Data/Data_and_task/B/X_train_observed.parquet')
X_observed_c = pd.read_parquet('../../Data/Data_and_task/C/X_train_observed.parquet')

#Loading estimated/forecasted test_weather from file
X_test_a = pd.read_parquet('../../Data/Data_and_task/A/X_test_estimated.parquet')
X_test_b = pd.read_parquet('../../Data/Data_and_task/B/X_test_estimated.parquet')
X_test_c = pd.read_parquet('../../Data/Data_and_task/C/X_test_estimated.parquet')

y_b = drop_repeating_sequences(y_b.copy())
y_b = delete_ranges_of_zeros_and_interrupting_values(y_b.copy(),200,[0.8625])
y_b = delete_ranges_of_zeros_and_interrupting_values(y_b.copy(),25,[0.8625])
y_b = drop_long_sequences(y_b.copy(),25)

X_estimated_a = add_all_features(X_estimated_a)
X_estimated_b = add_all_features(X_estimated_b)
X_estimated_c = add_all_features(X_estimated_c)

X_observed_a = add_all_features(X_observed_a)
X_observed_b = add_all_features(X_observed_b)
X_observed_c = add_all_features(X_observed_c)

X_test_a = add_all_features(X_test_a)
X_test_b = add_all_features(X_test_b)
X_test_c = add_all_features(X_test_c)

wanted_months = [3,4,5,6,7,8,9]

selected_features = ['date_forecast',
 'absolute_humidity_2m:gm3',
 'air_density_2m:kgm3',
 'ceiling_height_agl:m',
 'clear_sky_energy_1h:J',
 'clear_sky_rad:W',
 'cloud_base_agl:m',
 'dew_or_rime:idx',
 'dew_point_2m:K',
 'diffuse_rad:W',
 'diffuse_rad_1h:J',
 'direct_rad:W',
 'direct_rad_1h:J',
 'effective_cloud_cover:p',
 'elevation:m',
 'fresh_snow_12h:cm',
 'fresh_snow_1h:cm',
 'fresh_snow_24h:cm',
 'fresh_snow_3h:cm',
 'fresh_snow_6h:cm',
 'is_day:idx',
 'is_in_shadow:idx',
 'msl_pressure:hPa',
 'precip_5min:mm',
 'precip_type_5min:idx',
 'pressure_100m:hPa',
 'pressure_50m:hPa',
 'prob_rime:p',
 'rain_water:kgm2',
 'relative_humidity_1000hPa:p',
 'sfc_pressure:hPa',
 'snow_density:kgm3',
 'snow_depth:cm',
 'snow_drift:idx',
 'snow_melt_10min:mm',
 'snow_water:kgm2',
 'sun_azimuth:d',
 'sun_elevation:d',
 'super_cooled_liquid_water:kgm2',
 't_1000hPa:K',
 'total_cloud_cover:p',
 'visibility:m',
 'wind_speed_10m:ms',
 'wind_speed_u_10m:ms',
 'wind_speed_v_10m:ms',
 'wind_speed_w_1000hPa:ms',
 'dif_dat_rad',
 'hour',
 'minute',
 'month',
 'time_decimal',
 'hour_sin',
 'hour_cos']
selected_features.remove('ceiling_height_agl:m')
selected_features.remove('cloud_base_agl:m')
selected_features.remove('snow_density:kgm3')
