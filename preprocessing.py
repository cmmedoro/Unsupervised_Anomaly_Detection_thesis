# In this file we can find all the code related to the preprocessing step over the timeseries data
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler

def impute_nulls(dataframe):
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


# New time features: they are used to encode in a continuous way the time
def sin_transformer(period):
  return FunctionTransformer(lambda x: np.sin(x/period*2*np.pi))

def cos_transformer(period):
  return FunctionTransformer(lambda x: np.cos(x/period*2*np.pi))

def add_trigonometric_features(dataframe):
  dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
  dataframe['weekday']=dataframe['timestamp'].dt.weekday
  dataframe['weekday_y']=sin_transformer(7).fit_transform(dataframe['weekday'])
  dataframe['weekday_x']=cos_transformer(7).fit_transform(dataframe['weekday'])
  return dataframe

def impute_missing_dates(dataframe):
  """
  Take first and last timestamp available. Create a new index starting from these two values, making sure that the index is 
  sampled with 1 hour jump. Use ffill to impute the missing values for the dates newly created.
  """
  dataframe = dataframe.set_index(['timestamp'])
  start_ts = min(dataframe.index)
  end_ts = max(dataframe.index)
  new_index = pd.date_range(start_ts, end=end_ts, freq="1H")
  dfs_dict = {}
  for building_id, gdf in dataframe.groupby("building_id"):
      dfs_dict[building_id] = gdf.reindex(new_index, method='ffill')
  return dfs_dict

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

# sequences divided per building
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
def create_train_eval_sequences(dataframe, time_steps):
  scaler = MinMaxScaler(feature_range=(0,1))
  output = []
  output2=[]
  for building_id, gdf in dataframe.groupby("building_id"):
      gdf[['meter_reading', 'sea_level_pressure']] = scaler.fit_transform(gdf[['meter_reading', 'sea_level_pressure']])
      building_data = np.array(gdf[['meter_reading']]).astype(float) #, 'weekday_x', 'weekday_y', 'is_holiday'
      for i in range(len(building_data) - time_steps + 1):
        # find the end of this sequence
        end_ix = i + time_steps
        # check if we are beyond the dataset length for this building
        #if end_ix > len(building_data)-1:
         # break
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


# Define a function to revert the sliding window application
def reconstruction_windows(timeseries):
  """
  This function takes as input the windows regarding a single time series in the original dataset.
  Input: timeseries.shape = (8713, 72), where 72 = window_size, 8713 = number_of_windows = 24*366 - window_size + 1
  Output: media.shape = (24*366, 1) ---> meter_reading column reconstructed, by averaging the reconstructions for the same points
  """
  df_x_train = pd.DataFrame(np.squeeze(timeseries))
  transposed_df = df_x_train.T
  nan_rows = np.full((8784-transposed_df.shape[0],transposed_df.shape[1]),np.nan)
  rows = pd.DataFrame(nan_rows)
  transposed_df = pd.concat([transposed_df, rows], ignore_index = True) #transposed_df.append(rows, ignore_index = True)
  for i in range(0, transposed_df.shape[1]):
    col = transposed_df.iloc[:, i].shift(i)
    transposed_df.iloc[:, i] = col
  retransposed = transposed_df.T
  media = np.nanmean(retransposed, axis = 0)
  return media

def apply_reconstruction(dataframe, n_timeseries):
  # Take the entire dataframe
  squeezed_df = np.squeeze(dataframe)
  # Reshape it: ex. train (162, 8713, 72) ---> 162 = number of timeseries in the dataframe
  reshaped = np.reshape(squeezed_df, (n_timeseries, 8712, 72)) #8713
  reconstruction = []
  i = 0
  for timeseries in reshaped:
    # Reconstruct separately each timeseries
    print(i)
    average_reconstruction = reconstruction_windows(timeseries)
    reconstruction.append(average_reconstruction)
    i = i+1
  final_reconstruction = np.squeeze(reconstruction)
  fr = np.reshape(final_reconstruction, (n_timeseries*8784, 1))
  return fr
  

