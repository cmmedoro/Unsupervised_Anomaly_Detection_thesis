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
  
def padding_w(w, batch_size):
  # This function needs to be used on the outputs of the decoders: the last "batch" is not going to be full of the batch_size elements 
  # characterizing a single batch, therefore a proper padding needs to be ensured. Moreover, with this function, we also transform
  # the padded output into a suitable format to perform the following operations to obtain the reconstructed input
  last = w[-1]
  padded_last = F.pad(last, (0, 0, 0, batch_size-last.size()[0]))
  new_lista = w[:-1]
  new_lista.append(padded_last)
  res_w = torch.cat(new_lista, dim=0)
  index = batch_size-last.size()[0]
  original_rec = res_w[:-index]
  origin_rec = original_rec.detach().cpu().numpy()

  return origin_rec

##### RECONSTRUCTION WITH NON-OVERLAPPING WINDOWS #####
def get_wi_reconstructed(wi):
  reshaped_wi = [torch.flatten(wi_el) for wi_el in wi]
  stacked = torch.stack(reshaped_wi[:-1]).flatten()
  stacked_array = stacked.cpu().numpy()
  last_array = reshaped_wi[-1].cpu().numpy()
  total = np.concatenate([stacked_array, last_array])
  return total


##### ANOMALY DETECTION #####
def get_anomaly_dataset(test, total_w1, total_w2):
  scaler = MinMaxScaler(feature_range=(0,1))
  dfs_dict_1 = {}
  for building_id, gdf in test.groupby("building_id"):
      gdf[['meter_reading']]=scaler.fit_transform(gdf[['meter_reading']])
      dfs_dict_1[building_id] = gdf
  predicted_df_test = pd.concat(dfs_dict_1.values())
  predicted_df_test['reconstruction1'] = total_w1
  predicted_df_test['reconstruction2'] = total_w2
  return predicted_df_test

def define_threshold(predicted_df_test, wi): # wi can be 1 or 2
  col_loss = "relative_loss"+str(wi)
  th = "threshold"+str(wi)
  pred = "predicted_anomaly"+str(wi)
  thresholds=np.array([])
  for building_id, gdf in predicted_df_test.groupby("building_id"):
      val_mre_loss_building = gdf[col_loss].values
      building_threshold = (np.percentile(val_mre_loss_building, 75)) + 1.5 *((np.percentile(val_mre_loss_building, 75))-(np.percentile(val_mre_loss_building, 25)))
      gdf['threshold']=building_threshold
      thresholds= np.append(thresholds, gdf['threshold'].values)
  predicted_df_test[th]= thresholds
  predicted_df_test[pred] = predicted_df_test[col_loss] > predicted_df_test[th]
  predicted_df_test[pred]=predicted_df_test[pred].replace(False,0)
  predicted_df_test[pred]=predicted_df_test[pred].replace(True,1)
  return predicted_df_test

# Anomaly detection for overlapping windows
def get_overl_anomaly(train_window, test):
  scaler = MinMaxScaler(feature_range=(0,1))
  dfs_dict_1 = {}
  for building_id, gdf in test.groupby("building_id"):
    gdf[['meter_reading', 'sea_level_pressure']]=scaler.fit_transform(gdf[['meter_reading', 'sea_level_pressure']])
    dfs_dict_1[building_id] = gdf[train_window:]
  predicted_df = pd.concat(dfs_dict_1.values())
  return predicted_df

def define_overl_threshold(predicted_df_test, perc): # wi can be 1 or 2
  threshold = (np.percentile(predicted_df_test.anomaly_score.values, perc))
  predicted_df_test['threshold'] = threshold
  predicted_df_test['predicted_anomaly'] = predicted_df_test.anomaly_score > predicted_df_test['threshold']
  predicted_df_test['predicted_anomaly']=predicted_df_test['predicted_anomaly'].replace(False,0)
  predicted_df_test['predicted_anomaly']=predicted_df_test['predicted_anomaly'].replace(True,1)
  return predicted_df_test


### ANOMALY DETECTION FOR AUTOENCODERS ###
def get_predicted_dataset(dataset, reconstruction):
  # This will be used both for the test set and for the validation set, which is going to be used as reference for thresholds in some cases
  scaler = MinMaxScaler(feature_range=(0,1))
  dfs_dict_1 = {}
  for building_id, gdf in dataset.groupby("building_id"):
      gdf[['meter_reading']]=scaler.fit_transform(gdf[['meter_reading']])
      dfs_dict_1[building_id] = gdf
  predicted_df = pd.concat(dfs_dict_1.values())
  predicted_df['reconstruction'] = reconstruction
  predicted_df['abs_loss'] = np.abs(predicted_df.meter_reading - predicted_df.reconstruction)
  predicted_df['rel_loss'] = np.abs((predicted_df['predictions']-predicted_df['meter_reading'])/predicted_df['predictions'])
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
  threshold = threshold/val['meter_reading'].values[index]
  predicted_df['threshold'] = threshold
  predicted_df['predicted_anomaly'] = predicted_df['abs_loss'] > threshold * predicted_df['predictions']
  return predicted_df

def threshold_rel_loss(val, percentile, predicted_df):
  val_mre_loss = val['rel_loss'].values
  threshold = (np.percentile(np.squeeze(val_mre_loss), percentile))
  predicted_df['threshold'] = threshold
  predicted_df['predicted_anomaly'] = predicted_df['rel_loss'] > predicted_df['threshold']
  return threshold

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
    test_mre_loss_building= gdf['rel_loss'].testues
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


  
