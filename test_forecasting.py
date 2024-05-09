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
from utils_ae import *
from lstm import *

args = parser_file.parse_arguments()

model_type = args.model_type    

device = get_default_device()
#### Open the dataset ####
#energy_df = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/train.csv")
energy_df = pd.read_csv("/content/drive/MyDrive/ADSP/Backup_tesi_Carla_sorry_bisogno_di_gpu/train_features.csv")
# Select some columns from the original dataset
df = energy_df[['building_id','primary_use', 'timestamp', 'meter_reading', 'sea_level_pressure', 'is_holiday','anomaly', 'air_temperature']]

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
    residuals = pd.read_csv("/content/drive/MyDrive/ADSP/Backup_tesi_Carla_sorry_bisogno_di_gpu/residuals.csv")
    residui_df = residuals[['timestamp', 'building_id', 'primary_use', 'anomaly', 'meter_reading', 'sea_level_pressure', 'is_holiday', 'resid']]
    dfs_train, dfs_val, dfs_test = train_val_test_split(residui_df)
    train = pd.concat(dfs_train.values())
    val = pd.concat(dfs_val.values())
    test = pd.concat(dfs_test.values())
print(train.columns)
if args.do_multivariate:
    residuals = pd.read_csv("/content/drive/MyDrive/ADSP/Backup_tesi_Carla_sorry_bisogno_di_gpu/residuals.csv")
    residui_df = residuals[['timestamp', 'building_id', 'primary_use', 'anomaly', 'meter_reading', 'sea_level_pressure', 'is_holiday', 'resid']]
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
        output.append(seq_x)
        output2.append(seq_y)
  return np.stack(output), np.stack(output2)

### TRAINING THE MODEL ###
# For training we are going to create an input dataset consisting of overlapping windows of 72 measurements (3 days)
train_window = args.train_window

if args.do_multivariate:
    X_train, y_train = create_multivariate_train_eval_sequences(train, train_window)
    X_test, y_test = create_multivariate_train_eval_sequences(test, train_window)
    X_val, y_val = create_multivariate_train_eval_sequences(val, train_window)
else:
    X_train, y_train = create_train_eval_sequences(train, train_window)
    X_test, y_test = create_train_eval_sequences(test, train_window)
    X_val, y_val = create_train_eval_sequences(val, train_window)

BATCH_SIZE =  args.batch_size
N_EPOCHS = args.epochs
hidden_size = args.hidden_size

# batch_size, window_length, n_features = X_train.shape
batch, window_len, n_channels = X_train.shape

w_size = X_train.shape[1] * X_train.shape[2]
z_size = int(w_size * hidden_size) 
w_size, z_size


#val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().view(([X_val.shape[0], X_val.shape[1], X_val.shape[2]]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_test).float().view(([X_test.shape[0],X_test.shape[1], X_test.shape[2]]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)

z_size = 32
# Create the model and send it on the gpu device
model = LstmModel(n_channels, 32)
model = to_device(model, device)
print(model)

checkpoint_dir = args.checkpoint_dir
model = torch.load(checkpoint_dir)

results, forecast = testing(model, test_loader)
# On validation set
results_v, forecast_v = testing(model, val_loader)


reconstruction_test = np.concatenate([torch.stack(forecast[:-1]).flatten().detach().cpu().numpy(), forecast[-1].flatten().detach().cpu().numpy()])
reconstruction_val = np.concatenate([torch.stack(forecast_v[:-1]).flatten().detach().cpu().numpy(), forecast_v[-1].flatten().detach().cpu().numpy()])
    
predicted_df_val = get_predicted_dataset(val, reconstruction_val)
predicted_df_test = get_predicted_dataset(test, reconstruction_test)

threshold_method = args.threshold
percentile = args.percentile
weight_overall = args.weights_overall

predicted_df_test = anomaly_detection(predicted_df_val, predicted_df_test, threshold_method, percentile, weight_overall)

predicted_df_test.index.names=['timestamp']
predicted_df_test= predicted_df_test.reset_index()

predicted_df_test = pd.merge(predicted_df_test, df[['timestamp','building_id']], on=['timestamp','building_id'])

print(classification_report(predicted_df_test['anomaly'], predicted_df_test['predicted_anomaly']))
print(roc_auc_score(predicted_df_test['anomaly'], predicted_df_test['predicted_anomaly']))


print("Results based on the anomaly score")

# Qui va ad ottenere le label per ogni finestra
# Input modello è una lista di array, ognuno corrispondente a una sliding window con stride = 1 sui dati originali
# Quindi dobbiamo applicare la sliding window anche sulle label
windows_labels=[]
for b_id, gdf in test.groupby('building_id'):
    labels = gdf.anomaly.values
    for i in range(len(labels)-train_window):
        windows_labels.append(list(np.int_(labels[i:i+train_window])))
windows_labels

y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]

y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                            results[-1].flatten().detach().cpu().numpy()])

threshold=ROC(y_test,y_pred)
y_pred_ = np.zeros(y_pred.shape[0])
y_pred_[y_pred >= threshold] = 1
print(roc_auc_score(y_test, y_pred_))
print(classification_report(y_test, y_pred_))
print("OTHER METHOD: ")
threshold = np.percentile(y_pred, 80)
y_pred_ = np.zeros(y_pred.shape[0])
y_pred_[y_pred >= threshold] = 1
print(classification_report(y_test, y_pred_))
print(roc_auc_score(y_test, y_pred_))



