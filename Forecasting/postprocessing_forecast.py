import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import torch

# Define a function to turn possible anomalies in correspondance of original missing values into non anomalies
# Note: in the dataset, all the meter readings having NaN are associated to 0 (not an anomaly), so the predictions should be accustumed to this
def postprocessing_on_missing_values(dataframe_original, dataframe_post_reconstruction):
  """
  The dataframe_original contains a column, "is_na", which indicates with a True if that row contained a NaN in correspondence of the 
  meter reading column. This is the
  Firstly, we define a merge between the two datasets, so that we already prune the missing dates that we imputed originally.
  Then, on the merged dataframe, we substitute all those anomalies which are not to be considered, as corresponded to NaN, into non-anomaly
  """
  corrected = []
  for i, row in dataframe_post_reconstruction.iterrows():
    if row.is_na == True:
      corrected.append(0)
    else:
      corrected.append(row.predicted_anomaly)
  dataframe_post_reconstruction.predicted_anomaly = corrected
  #dataframe_post_reconstruction = pd.merge(dataframe_post_reconstruction, dataframe_original[['timestamp', 'building_id']], on = ['timestamp', 'building_id'])
  return dataframe_post_reconstruction



### ANOMALY DETECTION FOR LSTM FORECASTING (UNIVARIATE) ###
def get_predicted_dataset(dataset, forecast, train_window):
  # This will be used both for the test set and for the validation set, which is going to be used as reference for thresholds in some cases
  scaler = MinMaxScaler(feature_range=(0,1))
  dfs_dict_1 = {}
  for building_id, gdf in dataset.groupby("building_id"):
      gdf[['diff_lag_-1']]=scaler.fit_transform(gdf[['diff_lag_-1']])
      dfs_dict_1[building_id] = gdf[train_window:]
  predicted_df = pd.concat(dfs_dict_1.values())
  predicted_df['forecast'] = forecast
  predicted_df['abs_loss'] = np.abs(predicted_df.meter_reading - predicted_df.forecast)
  predicted_df['rel_loss'] = np.abs((predicted_df['forecast']-predicted_df['diff_lag_-1'])/predicted_df['forecast'])
  return predicted_df

def threshold_abs_loss(val, percentile, predicted_df):
  val_mae_loss = val['abs_loss'].values
  threshold = (np.percentile(val_mae_loss, percentile)) 
  predicted_df['threshold'] = threshold
  predicted_df['predicted_anomaly'] = predicted_df['abs_loss'] > predicted_df['threshold']
  return predicted_df

def threshold_norm_abs_loss(val, percentile, predicted_df):
  val_mae_loss = val['abs_loss'].values
  threshold = (np.percentile(val_mae_loss, percentile))
  difference_array = np.absolute(val_mae_loss - threshold)
  # find the index of minimum element from the array
  index = difference_array.argmin()
  #indexes = np.argmax(val_mae_loss)
  threshold = threshold/val['diff_lag_-1'].values[index]
  predicted_df['threshold'] = threshold
  predicted_df['predicted_anomaly'] = predicted_df['abs_loss'] > threshold * predicted_df['forecast']
  return predicted_df

def threshold_rel_loss(val, percentile, predicted_df):
  val_mre_loss = val['rel_loss'].values
  threshold = (np.percentile(np.squeeze(val_mre_loss), percentile))
  predicted_df['threshold'] = threshold
  predicted_df['predicted_anomaly'] = predicted_df['rel_loss'] > predicted_df['threshold']
  return predicted_df

def threshold_iqr_rel_loss(val, predicted_df):
  #Loss relativa
  val_mre_loss = val['rel_loss'].values
  #Interquartile threshold
  threshold = (np.percentile(np.squeeze(val_mre_loss), 75)) + 1.5 *((np.percentile(np.squeeze(val_mre_loss), 75))-(np.percentile(np.squeeze(val_mre_loss), 25)))
  predicted_df['threshold'] = threshold
  predicted_df['predicted_anomaly'] = predicted_df['rel_loss'] > predicted_df['threshold']
  return predicted_df

def threshold_iqr_test(predicted_df):
  #calculate threshold on relative loss quartiles but only on test, and in this case per building
  thresholds=np.array([])
  for building_id, gdf in predicted_df.groupby("building_id"):
    test_mre_loss_building= gdf['rel_loss'].values
    building_threshold = (np.percentile(np.squeeze(test_mre_loss_building), 75)) + 1.5 *((np.percentile(np.squeeze(test_mre_loss_building), 75))-(np.percentile(np.squeeze(test_mre_loss_building), 25)))
    gdf['threshold']=building_threshold
    thresholds= np.append(thresholds, gdf['threshold'].values)
  print(thresholds.shape)
  predicted_df['threshold']= thresholds
  predicted_df['predicted_anomaly'] = predicted_df['rel_loss'] > predicted_df['threshold']
  return predicted_df

def threshold_ewma(predicted_df):
  ewma =np.array([])
  for building_id, gdf in predicted_df.groupby("building_id"):
    test_mre_loss_building= gdf['rel_loss']
    gdf['ewma']=test_mre_loss_building.ewm(halflife=24, adjust=True).mean()#alpha=1#com=0.5
    ewma = np.append(ewma , gdf['ewma'].values)
  print(ewma.shape)
  predicted_df['ewma']= ewma
  predicted_df['difference']= np.abs(predicted_df['ewma']- predicted_df['rel_loss'])
  predicted_df['threshold'] = (np.percentile(np.squeeze(predicted_df['difference'].values), 75)) + 1.5 *((np.percentile(np.squeeze(predicted_df['difference'].values), 75))-(np.percentile(np.squeeze(predicted_df['difference'].values), 25)))

  predicted_df['predicted_anomaly'] = predicted_df['difference'] > predicted_df['threshold']
  return predicted_df

def threshold_weighted_rel_loss_iqr(predicted_df_val, predicted_df_test, weight_overall):
  val_mre_loss = predicted_df_val['rel_loss'].values
  threshold = (np.percentile(np.squeeze(val_mre_loss), 75)) + 1.5 *((np.percentile(np.squeeze(val_mre_loss), 75))-(np.percentile(np.squeeze(val_mre_loss), 25)))
  overall_threshold = threshold
  weighted_threshold =np.array([])
  for building_id, gdf in predicted_df_test.groupby("building_id"):
    building_threshold=(np.percentile(np.squeeze(gdf['rel_loss'].values), 75)) + 1.5 *((np.percentile(np.squeeze(gdf['rel_loss'].values), 75))-(np.percentile(np.squeeze(gdf['rel_loss'].values), 25)))
    new_threshold = weight_overall * overall_threshold + (1-weight_overall) * building_threshold
    gdf['weighted_threshold'] = new_threshold
    weighted_threshold = np.append(weighted_threshold , gdf['weighted_threshold'].values)
  predicted_df_test['weighted_threshold']=weighted_threshold
  predicted_df_test['predicted_anomaly'] = predicted_df_test['rel_loss'] > predicted_df_test['weighted_threshold']
  return predicted_df_test

def anomaly_detection(predicted_df_val, predicted_df_test, method_nr, percentile, weight_overall = 0.5):
  if method_nr == 0:
    predicted_df = threshold_abs_loss(predicted_df_val, percentile, predicted_df_test)
  elif method_nr == 1:
    predicted_df = threshold_norm_abs_loss(predicted_df_val, percentile, predicted_df_test)
  elif method_nr == 2:
    predicted_df = threshold_rel_loss(predicted_df_val, percentile, predicted_df_test)
  elif method_nr == 3:
    predicted_df = threshold_iqr_rel_loss(predicted_df_val, predicted_df_test)
  elif method_nr == 4:
    predicted_df = threshold_iqr_test(predicted_df_test)
  elif method_nr == 5:
    predicted_df = threshold_ewma(predicted_df_test)
  elif method_nr == 6:
    predicted_df = threshold_weighted_rel_loss_iqr(predicted_df_val, predicted_df_test, weight_overall)
  predicted_df['predicted_anomaly']=predicted_df['predicted_anomaly'].replace(False,0)
  predicted_df['predicted_anomaly']=predicted_df['predicted_anomaly'].replace(True,1)
  return predicted_df


  
