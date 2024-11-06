import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.nn.functional as F
import torch


### ANOMALY DETECTION FOR AUTOENCODERS (UNIVARIATE) ###
def get_predicted_dataset(test, reconstruction):
  test['reconstruction'] = reconstruction
  test['abs_loss'] = np.abs(test.generation_kwh - test.reconstruction)
  test['rel_loss'] = np.abs((test['reconstruction']-test['generation_kwh'])/test['reconstruction'])
  return test

def get_predicted_dataset_big(test, reconstruction):
    scaler = MinMaxScaler(feature_range = (0,1))
    dict_test = {}
    for code, gdf in test.groupby('country_code'):
      gdf[['solar_generation_actual']] = scaler.fit_transform(gdf[['solar_generation_actual']])
      dict_test[code] = gdf
    predicted_df_test = pd.concat(dict_test.values())
    predicted_df_test['reconstruction'] = reconstruction
    predicted_df_test['abs_loss'] = np.abs(predicted_df_test.solar_generation_actual - predicted_df_test.reconstruction)
    predicted_df_test['rel_loss'] = np.abs((predicted_df_test['reconstruction']-predicted_df_test['solar_generation_actual'])/predicted_df_test['reconstruction'])
    return predicted_df_test

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
  threshold = threshold/val['generation_kwh'].values[index]
  predicted_df['threshold'] = threshold
  predicted_df['predicted_anomaly'] = predicted_df['abs_loss'] > threshold * predicted_df['reconstruction']
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
  test_mre_loss = predicted_df['rel_loss'].values
  threshold = (np.percentile(np.squeeze(test_mre_loss), 75)) + 1.5 *((np.percentile(np.squeeze(test_mre_loss), 75))-(np.percentile(np.squeeze(test_mre_loss), 25)))
  predicted_df['threshold']= threshold
  predicted_df['predicted_anomaly'] = predicted_df['rel_loss'] > predicted_df['threshold']
  return predicted_df

def threshold_ewma(predicted_df):
  test_mre_loss= predicted_df['rel_loss']
  predicted_df['ewma']=test_mre_loss.ewm(halflife=24, adjust=True).mean()#alpha=1#com=0.5
  predicted_df['difference']= np.abs(predicted_df['ewma']- predicted_df['rel_loss'])
  predicted_df['threshold'] = (np.percentile(np.squeeze(predicted_df['difference'].values), 75)) + 1.5 *((np.percentile(np.squeeze(predicted_df['difference'].values), 75))-(np.percentile(np.squeeze(predicted_df['difference'].values), 25)))

  predicted_df['predicted_anomaly'] = predicted_df['difference'] > predicted_df['threshold']
  return predicted_df

def threshold_weighted_rel_loss_iqr(predicted_df_val, predicted_df_test, weight_overall):
  val_mre_loss = predicted_df_val['rel_loss'].values
  threshold = (np.percentile(np.squeeze(val_mre_loss), 75)) + 1.5 *((np.percentile(np.squeeze(val_mre_loss), 75))-(np.percentile(np.squeeze(val_mre_loss), 25)))
  overall_threshold = threshold
  weighted_threshold = (np.percentile(np.squeeze(predicted_df_test['rel_loss'].values), 75)) + 1.5 *((np.percentile(np.squeeze(predicted_df_test['rel_loss'].values), 75))-(np.percentile(np.squeeze(predicted_df_test['rel_loss'].values), 25)))
  new_threshold = weight_overall * overall_threshold + (1-weight_overall) * weighted_threshold
  predicted_df_test['weighted_threshold']=new_threshold
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
"""
### CREATION OF DATASET FOR ANOMALY SCORE ASSESSMENT ###
def get_predicted_anomaly_score(dataset, pred_anomalies, new_anomalies):
  # This will be used both for the test set and for the validation set, which is going to be used as reference for thresholds in some cases
  scaler = MinMaxScaler(feature_range=(0,1))
  dfs_dict_1 = {}
  for building_id, gdf in dataset.groupby("building_id"):
      gdf[['meter_reading']]=scaler.fit_transform(gdf[['meter_reading']])
      upper_outlier_value=(np.percentile(gdf['meter_reading'].values, 75)) + 1.5 *((np.percentile(gdf['meter_reading'].values, 75))-(np.percentile(gdf['meter_reading'].values, 25)))
      lower_outlier_value = (np.percentile(gdf['meter_reading'].values, 25)) - 1.5 *((np.percentile(gdf['meter_reading'].values, 75))-(np.percentile(gdf['meter_reading'].values, 25)))
      gdf['outliers'] = [1 if (el<lower_outlier_value or el>upper_outlier_value) else 0 for el in gdf['meter_reading'].values]
      dfs_dict_1[building_id] = gdf
  predicted_df = pd.concat(dfs_dict_1.values())
  predicted_df['predictions'] = pred_anomalies
  predicted_df['new_anomalies'] = new_anomalies
  return predicted_df
"""

  
