from preprocessing import impute_missing_prod, split, create_sequences, split_big, create_sequences_big
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from postprocessing import *
import torch.utils.data as data_utils
import parser_file
import warnings
warnings.filterwarnings('ignore')

args = parser_file.parse_arguments()

model_type = args.model_type

if model_type == "linear_ae":
    from linear_ae import *
    from utils_ae import *
elif model_type == "conv_ae":
    from convolutional_ae import *
    from utils_ae import *
elif model_type == "lstm_ae":
    from lstm_ae import *
    from utils_ae import *


if torch.cuda.is_available():
    device =  torch.device('cuda')
else:
    device = torch.device('cpu')

#### Open the dataset ####
# Original dataset
#production_df = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/production_ts.csv")
production_df = pd.read_csv("/content/drive/MyDrive/Prova_Transformers_production/production_ts.csv")
production_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
# There are some countries with no values for solar generation, so we can drop them out
no_countries = ['DELU', 'HR', 'HU', 'PL']
only_prod_df = production_df[production_df.country_code.isin(no_countries) == False]

only_prod_df['datetime'] = pd.to_datetime(only_prod_df.datetime)
# Proceed with imputing the missing values
# NOTE: given that the time series have a periodic nature (production at zero during the night, then it increases and finally decreases), it makes sense to try to impute the values
# by considering replicating the previous 24 hours
final_prod_df = impute_missing_prod()


# Select some columns from the original dataset
final_prod_df1 = final_prod_df[['generation_kwh']]

### PREPROCESSING ###
# Split the dataset into train, validation and test
dfs_train, dfs_val, dfs_test = split_big(final_prod_df1)
train = dfs_train.reset_index(drop = True)
val = dfs_val.reset_index(drop = True)
test = dfs_test.reset_index(drop = True)

#scaler = MinMaxScaler(feature_range = (0,1))
#X_train = scaler.fit_transform(dfs_train)
#X_val = scaler.transform(dfs_val)
#X_test = scaler.transform(test)
### TRAINING THE MODEL ###
# For training we are going to create an input dataset consisting of overlapping windows of 72 measurements (3 days)
#### CAMBIA LE FEATURES DA TENERE IN CASO MULTIVARIATO
train_window = args.train_window
X_t = create_sequences_big(dfs_train, train_window)
X_v = create_sequences_big(dfs_val, train_window)
X_te = create_sequences_big(dfs_test, train_window, train_window)


BATCH_SIZE =  args.batch_size
N_EPOCHS = args.epochs
hidden_size = args.hidden_size

# batch_size, window_length, n_features = dfs_train.shape
batch, window_len, n_channels = X_t.shape

w_size = X_t.shape[1] * X_t.shape[2]
z_size = int(w_size * hidden_size) 

if model_type == "conv_ae" or model_type == "lstm_ae" :
    #Credo di dover cambiare X_train.shape[0], w_size, X_train.shape[2] con X_train.shape[0], X_train.shape[1], X_train.shape[2]
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_t).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_v).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
   
elif model_type == "linear_ae" and args.do_multivariate:
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_t).float().reshape(([X_t.shape[0], w_size])), torch.from_numpy(X_t).float().reshape(([X_t.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_v).float().view(([X_v.shape[0], w_size])), torch.from_numpy(X_v).float().view(([X_v.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
else:
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_t).float().reshape(([X_t.shape[0], w_size])), torch.from_numpy(X_t).float().reshape(([X_t.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_v).float().view(([X_v.shape[0], w_size])), torch.from_numpy(X_v).float().view(([X_v.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    
#if model_type == "lstm_ae" or model_type == "conv_ae":
 #   z_size = 32
# Create the model and send it on the gpu device
if model_type == "lstm_ae":
    model = LstmAE(n_channels, z_size, train_window)
elif model_type == "conv_ae":
    model = ConvAE(n_channels, z_size) #n_channels
elif model_type == "linear_ae":
    model = LinearAE(w_size, z_size)

print(device)
model = model.to(device) 
print(model)

# Start training
history = training(N_EPOCHS, model, train_loader, val_loader, device)
print(history)
  
#plot_history(history)
checkpoint_path = args.save_checkpoint_dir
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