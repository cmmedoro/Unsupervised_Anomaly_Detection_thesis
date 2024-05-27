from preprocessing_forecast import *
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
# 4) Add lag features and differences with respect to the meter_reading value
# N.B.: we are using the shift function of Dataframes ---> if we pass a positive number as lag it means that we are going to link each
# datapoint to a previous one, whereas when we pass a negative value we relate the datapoint to a future observation
lags = [-1, 24, -24, 168, -168]
df2 = create_diff_lag_features(df2, lags)

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
    residui_df = residuals[['timestamp', 'building_id', 'primary_use', 'anomaly', 'meter_reading', 'sea_level_pressure', 'is_holiday', 'resid', 'air_temperature']]
    dfs_train, dfs_val, dfs_test = train_val_test_split(residui_df)
    train = pd.concat(dfs_train.values())
    val = pd.concat(dfs_val.values())
    test = pd.concat(dfs_test.values())
print(train.columns)
if args.do_multivariate:
    residuals = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/residuals2.csv")
    residui_df = residuals[['timestamp', 'building_id', 'primary_use', 'anomaly', 'meter_reading', 'sea_level_pressure', 'is_holiday', 'resid', 'air_temperature']]
    residui_df = add_trig_resid(residui_df)
    # Add lag features and differences with respect to the meter_reading value
    # N.B.: we are using the shift function of Dataframes ---> if we pass a positive number as lag it means that we are going to link each
    # datapoint to a previous one, whereas when we pass a negative value we relate the datapoint to a future observation
    lags = [-1, 24, -24, 168, -168]
    residui_df = create_diff_lag_features(residui_df, lags)
    dfs_train, dfs_val, dfs_test = train_val_test_split(residui_df)
    train = pd.concat(dfs_train.values())
    val = pd.concat(dfs_val.values())
    test = pd.concat(dfs_test.values())
    train.columns

### TRAINING THE MODEL ###
# For training we are going to create an input dataset consisting of overlapping windows of 72 measurements (3 days)
train_window = args.train_window
if args.do_multivariate:
    X_train, y_train = create_multivariate_train_eval_sequences(train, train_window)
    X_val, y_val = create_multivariate_train_eval_sequences(val, train_window)
else:
    X_train, y_train = create_train_eval_sequences(train, train_window)
    X_val, y_val = create_train_eval_sequences(val, train_window)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

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
np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/history_lstm_forecasting_multi_16feats_27_05.npy', history_to_save)
#/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis
