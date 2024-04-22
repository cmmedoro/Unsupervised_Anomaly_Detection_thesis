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

device = get_default_device()

#### Open the dataset ####
# Original dataset
#energy_df = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/train.csv")
energy_df = pd.read_csv("/content/drive/MyDrive/Unsupervised_Anomaly_Detection_thesis/train_features.csv")
# Select some columns from the original dataset
df = energy_df[['building_id','primary_use', 'timestamp', 'meter_reading', 'sea_level_pressure', 'is_holiday','anomaly']]

### PREPROCESSING ###
# 1) Impute missing values
imputed_df = impute_nulls(df)
# 2) Add trigonometric features
df = add_trigonometric_features(imputed_df)
# 3) Resample the dataset: measurement frequency = "1h"
dfs_dict = impute_missing_dates(df)
df1 = pd.concat(dfs_dict.values())

# Split the dataset into train, validation and test
dfs_train, dfs_val, dfs_test = train_val_test_split(df1)
train = pd.concat(dfs_train.values())
val = pd.concat(dfs_val.values())
test = pd.concat(dfs_test.values())

if args.do_resid:
    # Residuals dataset (missing values and dates imputation already performed)
    #residuals = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/residuals2.csv")
    residuals = pd.read_csv("/content/drive/MyDrive/Unsupervised_Anomaly_Detection_thesis/residuals.csv")
    residui_df = residuals[['timestamp', 'building_id', 'primary_use', 'anomaly', 'meter_reading', 'sea_level_pressure', 'is_holiday', 'resid']]
    dfs_train, dfs_val, dfs_test = train_val_test_split(residui_df)
    train = pd.concat(dfs_train.values())
    val = pd.concat(dfs_val.values())
    test = pd.concat(dfs_test.values())
print(train.columns)
### TRAINING THE MODEL ###
# For training we are going to create an input dataset consisting of overlapping windows of 72 measurements (3 days)
train_window = args.train_window
X_train, y_train = create_train_eval_sequences(train, train_window)
X_val, y_val = create_train_eval_sequences(val, train_window)


BATCH_SIZE =  args.batch_size
N_EPOCHS = args.epochs
hidden_size = args.hidden_size

# batch_size, window_length, n_features = X_train.shape
batch, window_len, n_channels = X_train.shape

w_size = X_train.shape[1] * X_train.shape[2]
z_size = int(w_size * hidden_size) 

if model_type == "conv_ae" or model_type == "lstm_ae" or model_type == "usad_conv" or model_type == "usad_lstm":
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().view(([X_train.shape[0], w_size, 1]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().view(([X_val.shape[0],w_size, 1]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
else:
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().view(([X_train.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().view(([X_val.shape[0],w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

if model_type == "lstm_ae" or model_type == "conv_ae":
    z_size = 32
# Create the model and send it on the gpu device
if model_type == "lstm_ae":
    model = LstmAE(n_channels, z_size, train_window)
elif model_type == "conv_ae":
    model = ConvAE(n_channels, z_size)
elif model_type == "linear_ae":
    model = LinearAE(w_size, z_size)
else:
    model = UsadModel(w_size, z_size)

print(device)
model = to_device(model, device)
print(model)

# Start training
history = training(N_EPOCHS, model, train_loader, val_loader)
print(history)
if model_type == "lstm_ae" or model_type == "conv_ae":
    history_to_save = torch.stack(history).flatten().detach().cpu().numpy()
    np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/history_lstm.npy', history_to_save) #content/checkpoints er prove su drive
else:
    np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/history_lstm.npy', history)
#plot_history(history)
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
