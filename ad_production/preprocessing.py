# In this file we can find all the code related to the preprocessing step over the timeseries data
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch

### IMPUTING MISSING DATES ###
def impute_missing_dates(dataframe):
  """
  Take first and last timestamp available. Create a new index starting from these two values, making sure that the index is 
  sampled with 1 hour jump. Use ffill to impute the missing values for the dates newly created.
  """
  dataframe = dataframe.set_index(['datetime'])
  start_ts = min(dataframe.index)
  end_ts = max(dataframe.index)
  new_index = pd.date_range(start_ts, end=end_ts, freq="1H")
  new_df = dataframe.reindex(new_index, method = "ffill")
  return new_df

### ADDING NEW FEATURES ###

# New time features: they are used to encode in a continuous way the time
def sin_transformer(period):
  return FunctionTransformer(lambda x: np.sin(x/period*2*np.pi))

def cos_transformer(period):
  return FunctionTransformer(lambda x: np.cos(x/period*2*np.pi))

def add_trigonometric_features(dataframe):
  dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
  dataframe['weekday']=dataframe['timestamp'].dt.weekday
  dataframe['month'] = dataframe['timestamp'].dt.month
  dataframe['hour'] = dataframe['timestamp'].dt.hour
  #dataframe['weekday'] = dataframe.index.weekday
  dataframe['weekday_y']=sin_transformer(7).fit_transform(dataframe['weekday'])
  dataframe['weekday_x']=cos_transformer(7).fit_transform(dataframe['weekday'])
  dataframe['month_y']=sin_transformer(12).fit_transform(dataframe['month'])
  dataframe['month_x']=cos_transformer(12).fit_transform(dataframe['month'])
  dataframe['hour_y']=sin_transformer(24).fit_transform(dataframe['hour'])
  dataframe['hour_x']=cos_transformer(24).fit_transform(dataframe['hour'])
  return dataframe

def create_lag_features(data_frame, list_lags):
  # Create lag features: consider the column 'generation_kwh' and for each row of the dataset add the information on the previous x hours
  for i in list_lags:
    data_frame[f"lag_{i}"] = data_frame['generation_kwh'].shift(i)
  return data_frame

def create_diff_lag_features(dataframe, list_lags):
  df = create_lag_features(dataframe, list_lags)
  col_names = []
  for i in list_lags:
    col_names.append(f'lag_{i}')
  df.update(df[col_names].fillna(0))
  diff_cols = []
  for col in col_names:
    df[f'diff_{col}'] = df[col] - df['meter_reading']
    diff_cols.append(f'diff_{col}')
  df.update(df[diff_cols].fillna(0))
  return df

def add_rolling_feature(production_df, window=3):
    cols = ['generation_kwh']
    # Define rolling windows: here min_periods = minimum number of observations in window required to have a value
    rolled = production_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    lag_mdn = rolled.median().reset_index().astype(np.float16)
    for col in cols:
        production_df[f'{col}_mean_lag{window}'] = lag_mean[col]
        production_df[f'{col}_std_lag{window}'] = lag_std[col]
        production_df[f'{col}_mdn_lag{window}'] = lag_mdn[col]
        production_df.update(production_df[[f'{col}_mean_lag{window}', f'{col}_std_lag{window}', f'{col}_mdn_lag{window}']].fillna(0))
    return production_df

### SPLIT THE DATASET IN TRAIN, VAL, TEST ###

def split(dataframe):
  df_train1, df_test = train_test_split(dataframe, test_size = 0.2, shuffle = False)
  df_train, df_val = train_test_split(df_train1, test_Size = 0.2, shuffle = False)
  return df_train, df_val, df_test


# Generated training sequences to use in the model. Valid also for testing, where stride = time_steps
def create_sequences(dataframe, time_steps, stride = 1):
    sequences = []
    for i in range(0, len(dataframe) - time_steps + 1, stride):
        #end of sequence
        end_idx = i + time_steps
        if end_idx <= len(dataframe)+1:
            slice = dataframe.loc[i: (i + time_steps -1), :]
        sequences.append(slice)
    return np.stack(sequences)

# Prepare the dataset for the synthetic generation of anomalies
def synthetize_anomalies(df, ub_anomalies, indices_to_zero):
    df['synthetic_anomaly'] = np.zeros(len(df))
    df.loc[indices_to_zero, 'generation_kwh'] = 0
    df.loc[indices_to_zero, 'synthetic_anomaly'] = 1
    ub_an = ub_anomalies - len(indices_to_zero)
    test_copy = df.copy()
    test_copy.reset_index(inplace = True, drop = True)
    df = gaussian_noise_injection(test_copy, ub_an)
    return df

def gaussian_noise_injection(df, ub_anomalies):
    t_min = df.generation_kwh.min()
    t_max = df.generation_kwh.max()
    idx_list = np.random.choice(a=len(df), size=int(ub_anomalies), replace=False)
    count_an = 0
    for idx in idx_list:
        print(idx)
        if df.loc[idx, 'synthetic_anomaly'] == 0:
            q = df.loc[idx, 'generation_kwh'] + np.random.uniform(
                low=t_min,
                high=1.0 * (t_max - t_min),
            )
            df.loc[idx, 'generation_kwh'] = q
            df.loc[idx, 'synthetic_anomaly'] = 1
            count_an = count_an + 1
    return df



