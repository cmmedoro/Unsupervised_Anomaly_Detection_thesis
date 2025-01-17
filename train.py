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
import parser_file
import warnings
warnings.filterwarnings('ignore')

args = parser_file.parse_arguments()

model_type = args.model_type

if model_type == "usad":
    from USAD.usad import *
    from USAD.utils import *
elif model_type == "usad_conv":
    from USAD.usad_conv import *
    from USAD.utils import *
elif model_type == "usad_lstm":
    from USAD.usad_lstm import *
    from USAD.utils import *
elif model_type == "linear_ae":
    from linear_ae import *
    from utils_ae import *
elif model_type == "conv_ae":
    from convolutional_ae import *
    from utils_ae import *
elif model_type == "lstm_ae":
    from lstm_ae import *
    from utils_ae import *
elif model_type == "vae":
    from vae import *
    from utils_ae import *


if torch.cuda.is_available():
    device =  torch.device('cuda')
else:
    device = torch.device('cpu')

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

if args.do_resid:
    # Residuals dataset (missing values and dates imputation already performed)
    residuals = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/residuals2.csv")
    # ADSP/Backup_tesi_Carla_sorry_bisogno_di_gpu
    #residuals = pd.read_csv("/content/drive/MyDrive/ADSP/Backup_tesi_Carla_sorry_bisogno_di_gpu/residuals.csv")
    residui_df = residuals[['timestamp', 'building_id', 'primary_use', 'anomaly', 'meter_reading', 'sea_level_pressure', 'air_temperature', 'is_holiday', 'resid']]
    dfs_train, dfs_val, dfs_test = train_val_test_split(residui_df)
    train = pd.concat(dfs_train.values())
    val = pd.concat(dfs_val.values())
    test = pd.concat(dfs_test.values())
print(train.columns)
if args.do_multivariate:
    residuals = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/residuals2.csv")
    residui_df = residuals[['timestamp', 'building_id', 'primary_use', 'anomaly', 'meter_reading', 'sea_level_pressure', 'air_temperature', 'is_holiday', 'resid']]
    lags = [1, -1, 24, -24, 168, -168] #[1, -1]
    residui_df = create_diff_lag_features(residui_df, lags)
    residui_df = add_trig_resid(residui_df)
    residui_df = add_rolling_feature(residui_df, 12)
    residui_df = add_rolling_feature(residui_df, 24)
    dfs_train, dfs_val, dfs_test = train_val_test_split(residui_df)
    train = pd.concat(dfs_train.values())
    val = pd.concat(dfs_val.values())
    test = pd.concat(dfs_test.values())

### TRAINING THE MODEL ###
# For training we are going to create an input dataset consisting of overlapping windows of 72 measurements (3 days)
#### CAMBIA LE FEATURES DA TENERE IN CASO MULTIVARIATO
train_window = args.train_window
if args.do_multivariate:
    X_train, y_train = create_multivariate_train_eval_sequences(train, train_window)
    X_val, y_val = create_multivariate_train_eval_sequences(val, train_window)
else:
    #X_train, y_train = create_train_eval_sequences(train, train_window)
    #X_val, y_val = create_train_eval_sequences(val, train_window)
    X_train, y_train = create_sequences(train, train_window, 1)
    X_val, y_val = create_sequences(val, train_window, 1)


BATCH_SIZE =  args.batch_size
N_EPOCHS = args.epochs
hidden_size = args.hidden_size

# batch_size, window_length, n_features = X_train.shape
batch, window_len, n_channels = X_train.shape

w_size = X_train.shape[1] * X_train.shape[2]
z_size = int(w_size * hidden_size) 

if model_type == "conv_ae" or model_type == "lstm_ae" or model_type == "usad_conv" or model_type == "usad_lstm" or model_type == "vae":
    #Credo di dover cambiare X_train.shape[0], w_size, X_train.shape[2] con X_train.shape[0], X_train.shape[1], X_train.shape[2]
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().view(([X_train.shape[0], X_train.shape[1], X_train.shape[2]]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().view(([X_val.shape[0],X_train.shape[1], X_train.shape[2]]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
elif model_type == "linear_ae" and args.do_multivariate:
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().reshape(([X_train.shape[0], w_size])), torch.from_numpy(y_train).float().reshape(([y_train.shape[0], window_len]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().reshape(([X_val.shape[0], w_size])), torch.from_numpy(y_val).float().reshape(([y_val.shape[0], window_len]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
else:
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().reshape(([X_train.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().reshape(([X_val.shape[0],w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
if model_type == "lstm_ae" or model_type == "conv_ae" or model_type == "vae":
    z_size = 32
# Create the model and send it on the gpu device
if model_type == "lstm_ae":
    model = LstmAE(n_channels, z_size, train_window)
elif model_type == "conv_ae":
    model = ConvAE(n_channels, z_size) #n_channels
elif model_type == "linear_ae":
    model = LinearAE(w_size, z_size)
elif model_type == "vae":
    model = LstmVAE(n_channels, z_size, train_window)
else:
    model = UsadModel(w_size, z_size)

print(device)
model = model.to(device) #to_device(model, device)
print(model)

# Start training
history = training(N_EPOCHS, model, train_loader, val_loader, device)
print(history)
  
#plot_history(history)
checkpoint_path = args.save_checkpoint_dir
if model_type.startswith("usad"):
    torch.save({
        'encoder': model.encoder.state_dict(),
        'decoder1': model.decoder1.state_dict(),
        'decoder2': model.decoder2.state_dict()
        }, checkpoint_path) # the path should be set in the run.job file
else:
    torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder': model.decoder.state_dict()
            }, checkpoint_path) # the path should be set in the run.job file

#if model_type == "lstm_ae" or model_type == "conv_ae" or model_type == "vae":
 #   history_to_save = torch.stack(history).flatten().detach().cpu().numpy()
    #train_recos_to_save = np.concatenate([torch.stack(train_recos[:-1]).flatten().detach().cpu().numpy(), train_recos[-1].flatten().detach().cpu().numpy()])
    #train_recos_to_save = torch.stack(train_recos).flatten().detach().cpu().numpy()
    # /nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/history_lstm.npy
  #  np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/history_conv_ae_multi_outputMultiFeat_15_06.npy', history_to_save) #/content/checkpoints er prove su drive
    #np.save('/content/checkpoints/train_recos.npy', train_recos_to_save)
#else:
 #   np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/history_conv_ae_multi_outputMultiFeat_15_06.npy', history)
np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/history_linear_ae_multi_output!Feat_01_07.npy', history)