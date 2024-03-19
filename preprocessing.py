# In this file we can find all the code related to the preprocessing step over the timeseries data
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import FunctionTransformer

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
  dataframe= dataframe.set_index(['timestamp'])
  start_ts = min(dataframe.index)
  end_ts = max(dataframe.index)
  new_index = pd.date_range(start_ts, end=end_ts, freq="1H")
  dfs_dict = {}
  for building_id, gdf in dataframe.groupby("building_id"):
      dfs_dict[building_id] = gdf.reindex(new_index, method='ffill')
  dataframe = pd.concat(dfs_dict.values())

