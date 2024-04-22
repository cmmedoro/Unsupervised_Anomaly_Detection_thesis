from preprocessing import *
import preprocessing as prp
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
#from USAD.usad import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from postprocessing import *
import plotly.graph_objects as go
import torch.utils.data as data_utils
#from usad_conv import *
from lstm_ae import *
from utils_ae import *
import parser_file


device = get_default_device()

args = parser_file.parse_arguments()


#### Open the dataset ####
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

### TRAINING THE MODEL ###
# For training we are going to create an input dataset consisting of overlapping windows of 72 measurements (3 days)
train_window = args.train_window
X_train, y_train = create_train_eval_sequences(train, train_window)
X_val, y_val = create_train_eval_sequences(val, train_window)

if args.do_test:
    # Overlapping windows
    X_test, y_test = create_train_eval_sequences(test, train_window)
else:
    # Non-overlapping windows
    X_test, y_test = create_test_sequences(test, train_window)

BATCH_SIZE =  args.batch_size
N_EPOCHS = args.epochs
hidden_size = args.hidden_size

# batch_size, window_length, n_features = X_train.shape
batch, window_len, n_channels = X_train.shape

w_size = X_train.shape[1] * X_train.shape[2]
z_size = int(w_size * hidden_size) 
w_size, z_size

model_type = args.model_type

if model_type == "conv_ae"  or model_type == "lstm_ae":
    #train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().view(([X_train.shape[0], w_size, 1]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    #val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().view(([X_val.shape[0],w_size, 1]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_test).float().view(([X_test.shape[0],w_size, 1]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
else:
    #train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().view(([X_train.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    #val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().view(([X_val.shape[0],w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_test).float().view(([X_test.shape[0],w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

if model_type == "lstm_ae":
    z_size = 32
# Create the model and send it on the gpu device
model = LstmAE(n_channels, z_size, train_window)
model = to_device(model, device)
print(model)

if args.do_reconstruction:
    ### RECONSTRUCTING ###
    # Recover checkpoint
    checkpoint_dir = args.checkpoint_dir
    checkpoint = torch.load(checkpoint_dir)

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    #model.decoder2.load_state_dict(checkpoint['decoder2'])

    results, w = testing(model, test_loader)

    res_dir = args.res_dir

    reconstruction = np.concatenate([torch.stack(w[:-1]).flatten().detach().cpu().numpy(), w[-1].flatten().detach().cpu().numpy()])
  
    scaler = MinMaxScaler(feature_range=(0,1))
    dfs_dict_1 = {}
    for building_id, gdf in test.groupby("building_id"):
        gdf[['meter_reading']]=scaler.fit_transform(gdf[['meter_reading']])
        dfs_dict_1[building_id] = gdf
    predicted_df_test = pd.concat(dfs_dict_1.values())

    predicted_df_test['reconstruction'] = reconstruction

    predicted_df_test['relative_loss'] = np.abs((predicted_df_test['reconstruction']-predicted_df_test['meter_reading'])/predicted_df_test['reconstruction'])

    #calculate threshold on relative loss quartiles but only on val, and in this case per building
    thresholds=np.array([])
    for building_id, gdf in predicted_df_test.groupby("building_id"):
        val_mre_loss_building= gdf['relative_loss'].values
        building_threshold = (np.percentile(val_mre_loss_building, 75)) + 1.5 *((np.percentile(val_mre_loss_building, 75))-(np.percentile(val_mre_loss_building, 25)))
        gdf['threshold']=building_threshold
        thresholds= np.append(thresholds, gdf['threshold'].values)
    print(thresholds.shape)
    predicted_df_test['threshold']= thresholds

    predicted_df_test['predicted_anomaly'] = predicted_df_test['relative_loss'] > predicted_df_test['threshold']
    predicted_df_test['predicted_anomaly']=predicted_df_test['predicted_anomaly'].replace(False,0)
    predicted_df_test['predicted_anomaly']=predicted_df_test['predicted_anomaly'].replace(True,1)

    predicted_df_test.index.names=['timestamp']
    predicted_df_test= predicted_df_test.reset_index()

    predicted_df_test = pd.merge(predicted_df_test, df[['timestamp','building_id']], on=['timestamp','building_id'])

    print(classification_report(predicted_df_test['anomaly'], predicted_df_test['predicted_anomaly']))
    print(roc_auc_score(predicted_df_test['anomaly'], predicted_df_test['predicted_anomaly']))


elif args.do_test:
    ### TESTING ###
    # Recover checkpoint
    checkpoint_dir = args.checkpoint_dir
    checkpoint = torch.load(checkpoint_dir)

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    #model.decoder2.load_state_dict(checkpoint['decoder2'])

    results, w = testing(model,test_loader)
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



