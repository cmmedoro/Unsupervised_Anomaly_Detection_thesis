from preprocessing import *
import preprocessing as prp
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from postprocessing import *
import plotly.graph_objects as go
import torch.utils.data as data_utils
#import parser_file
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

#args = parser_file.parse_arguments()

#model_type = args.model_type


#### Open the dataset ####
# Original dataset
energy_df = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/train.csv")
#energy_df = pd.read_csv("/content/drive/MyDrive/ADSP/Backup_tesi_Carla_sorry_bisogno_di_gpu/train_features.csv")
# Select some columns from the original dataset
df = energy_df[['building_id','primary_use', 'timestamp', 'meter_reading', 'sea_level_pressure', 'is_holiday','anomaly', 'air_temperature']]

### PREPROCESSING ###
# 1) Impute missing values
imputed_df = impute_nulls(df)
# 2) Resample the dataset: measurement frequency = "1h"
dfs_dict = impute_missing_dates(imputed_df)
df1 = pd.concat(dfs_dict.values())
# 3) Add trigonometric features
df2 = add_trigonometric_features(df1)

lags = [1, -1]
df2 = create_diff_lag_features(df2, lags)

# Split the dataset into train, validation and test
dfs_train, dfs_val, dfs_test = train_val_test_split(df2)
train = pd.concat(dfs_train.values())
val = pd.concat(dfs_val.values())
test = pd.concat(dfs_test.values())

def create_train_eval_sequences(dataframe, time_steps):
  scaler = MinMaxScaler(feature_range=(0,1))
  output = []
  output2=[]
  for building_id, gdf in dataframe.groupby("building_id"):
      gdf[['meter_reading']] = scaler.fit_transform(gdf[['meter_reading']])
      building_data = np.array(gdf[['meter_reading']]).astype(float)
      for i in range(len(building_data)):
        # find the end of this sequence
        end_ix = i + time_steps
        # check if we are beyond the dataset length for this building
        if end_ix > len(building_data)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = building_data[i:end_ix, :], building_data[end_ix, 0]
        output.append(seq_x)
        output2.append(seq_y)
  return np.stack(output), np.stack(output2)

#X_train, y_train = create_train_eval_sequences(train, 168)
#X_val, y_val = create_train_eval_sequences(val, 168)
#X_test, y_test = create_train_eval_sequences(test, 168)

#1249, 1259, 1264, 1279

times = test[test.building_id == 1279]
print(1279)
x_t, y_t = create_train_eval_sequences(times, 168)
x_t.shape, y_t.shape


next_timestamp_test = []
for window in x_t:
  arima = ARIMA(window, order = (1, 0, 0))
  model_fit = arima.fit()
  prediction = model_fit.forecast(steps = 1)
  next_timestamp_test.append(prediction[0])

np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/outputs/predictions_test_1279.npy', next_timestamp_test)
