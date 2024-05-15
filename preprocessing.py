# In this file we can find all the code related to the preprocessing step over the timeseries data
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import torch.nn.functional as F
import torch

### IMPUTATION OF NULL VALUES ###

def impute_nulls(dataframe):
    # Keep track of nan values for postprocessing
    dataframe['is_na'] = dataframe.meter_reading.isna()
    # Sort the dataframe
    sorted_df = dataframe.sort_values(by = ['building_id', 'timestamp'])
    # We pass to lists because it is easier than dealing with Series objects
    buildings = sorted_df.building_id.tolist()
    readings = sorted_df.meter_reading.values
    result = [None if np.isnan(el) == True else el for el in readings]
    # Consider the mean of 'meter_reading' for each building id: this would be useful when imputing missing values, in particular when we have NaN values at the beginning or at the end
    # of a time series
    averages = sorted_df.groupby(['building_id'])['meter_reading'].mean()
    fill_gaps(result, buildings, averages)
    sorted_df.meter_reading = result
    return sorted_df


def fill_gaps(readings, buildings, averages):
  right_index = 0
  right_value = 0
  for i, value in enumerate(readings):
    if i == 0 or buildings[i] != buildings[i-1]: # If we are at the beginning, or we are changin buildings
      left_value = averages[buildings[i]]
    else:
      left_value = readings[i-1] # Take previous value
    # Reuse the right_value value, useful when there are multiple consecutive missing values
    if i < right_index:
        readings[i] = (left_value + right_value) / 2
        continue
    # Obtain the right value
    if buildings[i] != buildings[i-1]:
      right_value = averages[buildings[i]]
    if value == None:
      for j in range(i+1, len(readings)):
        if buildings[j] != buildings[i]: # Check whether the next buildings are different: if so, the right value is going to be 0
          right_value = averages[buildings[i]]
          break
        elif readings[j] != None:
          right_value = float(readings[j])
          break
      if i == len(readings)-1: # Edge case: the last value of the last building is empty
        right_value = averages[buildings[i]]
      right_index = j
      readings[i] = (left_value + right_value) / 2
    else:
      readings[i] = float(readings[i]) # Parse to float all present values

### IMPUTATION MISSING DATES ###
def impute_missing_dates(dataframe):
  """
  Take first and last timestamp available. Create a new index starting from these two values, making sure that the index is 
  sampled with 1 hour jump. Use ffill to impute the missing values for the dates newly created.
  """
  dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
  dataframe = dataframe.set_index(['timestamp'])
  start_ts = min(dataframe.index)
  end_ts = max(dataframe.index)
  new_index = pd.date_range(start_ts, end=end_ts, freq="1H")
  dfs_dict = {}
  for building_id, gdf in dataframe.groupby("building_id"):
      dfs_dict[building_id] = gdf.reindex(new_index, method='ffill')
  return dfs_dict

### ADDING NEW FEATURES ###

# New time features: they are used to encode in a continuous way the time
def sin_transformer(period):
  return FunctionTransformer(lambda x: np.sin(x/period*2*np.pi))

def cos_transformer(period):
  return FunctionTransformer(lambda x: np.cos(x/period*2*np.pi))

def add_trigonometric_features(dataframe):
  #dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
  #dataframe['weekday']=dataframe['timestamp'].dt.weekday
  dataframe['weekday'] = dataframe.index.weekday
  dataframe['weekday_y']=sin_transformer(7).fit_transform(dataframe['weekday'])
  dataframe['weekday_x']=cos_transformer(7).fit_transform(dataframe['weekday'])
  return dataframe

def add_trig_resid(dataframe):
  dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
  dataframe['weekday']=dataframe['timestamp'].dt.weekday
  #dataframe['weekday'] = dataframe.index.weekday
  dataframe['weekday_y']=sin_transformer(7).fit_transform(dataframe['weekday'])
  dataframe['weekday_x']=cos_transformer(7).fit_transform(dataframe['weekday'])
  return dataframe

def create_lag_features(data_frame, list_lags):
  # Create lag features: consider the column 'meter_reading' and for each row of the dataset add the information on the previous x hours
  # This should be done by buildings: for each timeseries, meaning for each building, we need to perform this operation, which means that the first measurements are going
  # to have some NaN values due to the fact that they have no x previous measurement
  buildings = data_frame['building_id'].unique()
  iterations = 0
  for bid in buildings:
    iterations += 1
    df = data_frame[data_frame['building_id'] == bid]
    for i in list_lags:
      df[f'lag_{i}'] = df['meter_reading'].shift(i)
    if iterations == 1:
      df_final = df
    else:
      df_final = pd.concat([df_final, df])
  return df_final

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

### SPLIT THE DATASET IN TRAIN, VAL, TEST ###

def split(dataframe, buildings_id):
  dfs_dict_1 = {}
  for building_id, gdf in dataframe.groupby("building_id"):
      if building_id in buildings_id:
        dfs_dict_1[building_id] = gdf
  return dfs_dict_1

def train_val_split(dataframe):
  building_ids_train = np.unique(dataframe.building_id)
  building_ids_train =[building_id for building_id in building_ids_train if building_id%5<4]

  dfs_dict_train = split(dataframe, building_ids_train)

  building_ids_val = np.unique(dataframe.building_id)
  building_ids_val =[building_id for building_id in building_ids_val if building_id%5==4]

  dfs_dict_val = split(dataframe, building_ids_val)

  return dfs_dict_train, dfs_dict_val


def train_val_test_split(dataframe):
  building_ids_temp = np.unique(dataframe.building_id)
  building_ids_temp =[building_id for building_id in building_ids_temp if building_id%5<4]

  building_ids_train = [building_id for building_id in building_ids_temp if building_id <= 1240]
  building_ids_val = [building_id for building_id in building_ids_temp if building_id > 1240]

  dfs_dict_train = split(dataframe, building_ids_train)
  dfs_dict_val = split(dataframe, building_ids_val)

  building_ids_test = np.unique(dataframe.building_id)
  building_ids_test =[building_id for building_id in building_ids_test if building_id%5==4]

  dfs_dict_test = split(dataframe, building_ids_test)

  return dfs_dict_train, dfs_dict_val, dfs_dict_test

# sequences divided per building: this is useful for FORECASTING
def split_sequences(dataframe, n_steps):
  scaler = MinMaxScaler(feature_range=(0,1))
  X, y = list(), list()
  for building_id, gdf in dataframe.groupby("building_id"):
    gdf[['meter_reading', 'sea_level_pressure']] = scaler.fit_transform(gdf[['meter_reading', 'sea_level_pressure']]) #,'weekday_x', 'weekday_y'
    building_data = np.array(gdf[['meter_reading']]).astype(float)#can choose to add additional features: 'sea_level_pressure', 'weekday_x', 'weekday_y', 'is_holiday'
    for i in range(len(building_data)):
      # find the end of this sequence
      end_ix = i + n_steps
      # check if we are beyond the dataset length for this building
      if end_ix > len(building_data)-1:
        break
      # gather input and output parts of the pattern
      seq_x, seq_y = building_data[i:end_ix, :], building_data[end_ix, 0]
      X.append(seq_x)
      y.append(seq_y)
  return np.array(X), np.array(y)



# Generated training sequences to use in the model.
def create_sequences(dataframe, time_steps, stride = 1):
  scaler = MinMaxScaler(feature_range=(0,1))
  output = []
  output2=[]
  for building_id, gdf in dataframe.groupby("building_id"):
      gdf[['meter_reading', 'sea_level_pressure']] = scaler.fit_transform(gdf[['meter_reading', 'sea_level_pressure']])
      building_data = np.array(gdf[['meter_reading']]).astype(float) #, 'weekday_x', 'weekday_y', 'is_holiday'
      for i in range(0, len(building_data) - time_steps + 1, stride): #range(0, len(building_data) - time_steps +1, stride):
        # find the end of this sequence
        end_ix = i + time_steps
        # check if we are beyond the dataset length for this building
        if end_ix > len(building_data)-1:
          break
        output.append(building_data[i : (i + time_steps),:])
        output2.append(building_data[i : (i + time_steps),0])
  return np.stack(output), np.stack(output2)


def create_train_eval_sequences(dataframe, time_steps):
  scaler = MinMaxScaler(feature_range=(0,1))
  output = []
  output2=[]
  for building_id, gdf in dataframe.groupby("building_id"):
      gdf[['meter_reading', 'sea_level_pressure']] = scaler.fit_transform(gdf[['meter_reading', 'sea_level_pressure']])
      building_data = np.array(gdf[['meter_reading']]).astype(float) #, 'weekday_x', 'weekday_y', 'is_holiday'
      for i in range(len(building_data) - time_steps + 1): #range(0, len(building_data) - time_steps +1, stride):
        # find the end of this sequence
        end_ix = i + time_steps
        # check if we are beyond the dataset length for this building
        if end_ix > len(building_data)-1:
          break
        output.append(building_data[i : (i + time_steps),:])
        output2.append(building_data[i : (i + time_steps),0])
  return np.stack(output), np.stack(output2)

def create_multivariate_train_eval_sequences(dataframe, time_steps):
  scaler = MinMaxScaler(feature_range=(0,1))
  output = []
  output2=[]
  for building_id, gdf in dataframe.groupby("building_id"):
      #'building_id','primary_use', 'timestamp', 'meter_reading', 'sea_level_pressure', 'is_holiday','anomaly', 'air_temperature'
      gdf[['meter_reading', 'sea_level_pressure', 'air_temperature', 'weekday_x', 'weekday_y', 'resid']] = scaler.fit_transform(gdf[['meter_reading', 'sea_level_pressure', 'air_temperature', 'weekday_x', 'weekday_y', 'resid']])
      building_data = np.array(gdf[['meter_reading', 'sea_level_pressure', 'air_temperature', 'is_holiday', 'weekday_x', 'weekday_y', 'resid']]).astype(float) 
      for i in range(len(building_data) - time_steps + 1):
        # find the end of this sequence
        end_ix = i + time_steps
        # check if we are beyond the dataset length for this building
        if end_ix > len(building_data)-1:
          break
        output.append(building_data[i : (i + time_steps),:])
        output2.append(building_data[i : (i + time_steps),0])
  return np.stack(output), np.stack(output2)

# Generated testing sequences for use in the model.
def create_test_sequences(dataframe, time_steps):
    scaler = MinMaxScaler(feature_range=(0,1))
    output = []
    output2 = []
    for building_id, gdf in dataframe.groupby("building_id"):
       gdf[['meter_reading', 'sea_level_pressure']] = scaler.fit_transform(gdf[['meter_reading', 'sea_level_pressure']])
       building_data = np.array(gdf[['meter_reading']]).astype(float) #, 'weekday_x', 'weekday_y', 'is_holiday'
       for i in range(0, len(building_data) - time_steps + 1, time_steps):
        end_ix = i + time_steps
        #if end_ix > len(gdf):
          #break
        output.append(building_data[i : end_ix,:])
        output2.append(building_data[i : end_ix,0])
    return np.stack(output), np.stack(output2)

def create_multivariate_test_sequences(dataframe, time_steps):
    scaler = MinMaxScaler(feature_range=(0,1))
    output = []
    output2 = []
    for building_id, gdf in dataframe.groupby("building_id"):
      gdf[['meter_reading', 'sea_level_pressure', 'air_temperature', 'weekday_x', 'weekday_y', 'resid']] = scaler.fit_transform(gdf[['meter_reading', 'sea_level_pressure', 'air_temperature', 'weekday_x', 'weekday_y', 'resid']])
      building_data = np.array(gdf[['meter_reading', 'sea_level_pressure', 'air_temperature', 'is_holiday', 'weekday_x', 'weekday_y', 'resid']]).astype(float) 
      for i in range(0, len(building_data) - time_steps + 1, time_steps):
        end_ix = i + time_steps
        #if end_ix > len(gdf):
          #break
        output.append(building_data[i : end_ix,:])
        output2.append(building_data[i : end_ix,0])
    return np.stack(output), np.stack(output2)


