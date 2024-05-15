from preprocessing import *
import preprocessing as prp
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
import plotly.graph_objects as go
import torch.utils.data as data_utils
import parser_file
import warnings
warnings.filterwarnings('ignore')
from utils_ae import *
from lstm import *

args = parser_file.parse_arguments()

model_type = args.model_type    

device = get_default_device()

#### Open the dataset ####
print("Inizio")
# Original dataset
#ADSP/Backup_tesi_Carla_sorry_bisogno_di_gpu
energy_df = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/train.csv")
#energy_df = pd.read_csv("/content/drive/MyDrive/Unsupervised_Anomaly_Detection_thesis/train_features.csv")
# Select some columns from the original dataset
df = energy_df[['building_id','primary_use', 'timestamp', 'meter_reading', 'sea_level_pressure', 'is_holiday','anomaly', 'air_temperature']]

### PREPROCESSING ###
# 1) Impute missing values
imputed_df = impute_nulls(df)

# 2) Resample the dataset: measurement frequency = "1h"
dfs_dict = impute_missing_dates(imputed_df)
df1 = pd.concat(dfs_dict.values())
#lags = [1, -1]
#df1 = create_diff_lag_features(df1, lags)
# 3) Add trigonometric features
df2 = add_trigonometric_features(df1)

# Split the dataset into train, validation and test
dfs_train, dfs_val, dfs_test = train_val_test_split(df2)
train = pd.concat(dfs_train.values())
val = pd.concat(dfs_val.values())
test = pd.concat(dfs_test.values())

if args.do_resid:
    # Residuals dataset (missing values and dates imputation already performed)
    residuals = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/residuals2.csv")
    # ADSP/Backup_tesi_Carla_sorry_bisogno_di_gpu
    #residuals = pd.read_csv("/content/drive/MyDrive/Unsupervised_Anomaly_Detection_thesis/residuals.csv")
    residui_df = residuals[['timestamp', 'building_id', 'primary_use', 'anomaly', 'meter_reading', 'sea_level_pressure', 'is_holiday', 'resid']]
    dfs_train, dfs_val, dfs_test = train_val_test_split(residui_df)
    train = pd.concat(dfs_train.values())
    val = pd.concat(dfs_val.values())
    test = pd.concat(dfs_test.values())
print(train.columns)
if args.do_multivariate:
    residuals = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/residuals2.csv")
    residui_df = residuals[['timestamp', 'building_id', 'primary_use', 'anomaly', 'meter_reading', 'sea_level_pressure', 'is_holiday', 'resid']]
    residui_df = add_trig_resid(residui_df)
    dfs_train, dfs_val, dfs_test = train_val_test_split(residui_df)
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

def create_multivariate_train_eval_sequences(dataframe, time_steps):
  scaler = MinMaxScaler(feature_range=(0,1))
  output = []
  output2=[]
  for building_id, gdf in dataframe.groupby("building_id"):
      gdf[['meter_reading', 'sea_level_pressure', 'air_temperature', 'weekday_x', 'weekday_y']] = scaler.fit_transform(gdf[['meter_reading', 'sea_level_pressure', 'air_temperature','weekday_x', 'weekday_y']])
      building_data = np.array(gdf[['meter_reading', 'sea_level_pressure', 'air_temperature', 'weekday_x', 'weekday_y']]).astype(float) 
      for i in range(len(building_data)):
        # find the end of this sequence
        end_ix = i + time_steps
        # check if we are beyond the dataset length for this building
        if end_ix > len(building_data)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = building_data[i:end_ix, :], building_data[end_ix, 0]
        # In seq_x we store the data for the window
        # In seq_y we store the meter_reading corresponding to the following data point in the sequence. This is the ground truth
        # we are going to use to compare the predictions made by the model
        output.append(seq_x)
        output2.append(seq_y)
  return np.stack(output), np.stack(output2)

### TRAINING THE MODEL ###
# For training we are going to create an input dataset consisting of overlapping windows of 72 measurements (3 days)
train_window = args.train_window
if args.do_multivariate:
    X_train, y_train = create_multivariate_train_eval_sequences(train, train_window)
    X_val, y_val = create_multivariate_train_eval_sequences(val, train_window)
else:
    X_train, y_train = create_train_eval_sequences(train, train_window)
    X_val, y_val = create_train_eval_sequences(val, train_window)


BATCH_SIZE =  args.batch_size
N_EPOCHS = args.epochs
hidden_size = args.hidden_size

# batch_size, window_length, n_features = X_train.shape
batch, window_len, n_channels = X_train.shape

w_size = X_train.shape[1] * X_train.shape[2]
z_size = int(w_size * hidden_size) 

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
#train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().view(([X_train.shape[0], X_train.shape[1], X_train.shape[2]]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
#labels_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.torch.from_numpy(y_train).float().view(([y_train.shape[0], y_train.shape[1], y_train.shape[2]]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
#val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().view(([X_val.shape[0],X_train.shape[1], X_train.shape[2]]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = LstmModel(n_channels, 32)


print(device)
model = to_device(model, device)
print(model)

# Start training
history = training(N_EPOCHS, model, train_loader, val_loader)
print(history)


#plot_history(history)
checkpoint_path = args.save_checkpoint_dir
torch.save(model.state_dict(), checkpoint_path)

history_to_save = torch.stack(history).flatten().detach().cpu().numpy()
np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/history_lstm_forecasting.npy', history_to_save)
#/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis
