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
#from usad_conv import *
import parser_file
from linear_ae import *
#from convolutional_ae import *
#from lstm_ae import *
#from Forecasting.lstm import *
from utils_ae import ROC
import warnings
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device =  torch.device('cuda')
else:
    device = torch.device('cpu')
#device = get_default_device()
#device = torch.device("cuda")

print(device)

args = parser_file.parse_arguments()
model_type = args.model_type
#Read data
normal = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/SWaT_Dataset_Normal_v1.csv")
normal = normal.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)

# Transform all columns into float64
for i in list(normal): 
    normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
normal = normal.astype(float)

# Normalization
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

x = normal.values
x_scaled = min_max_scaler.fit_transform(x)
normal = pd.DataFrame(x_scaled)

#Read data
attack = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/SWaT_Dataset_Attack_v0.csv",sep=";")
labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
attack = attack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)

# Transform all columns into float64
for i in list(attack):
    attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
attack = attack.astype(float)

# Normalization
from sklearn import preprocessing

x = attack.values 
x_scaled = min_max_scaler.transform(x)
attack = pd.DataFrame(x_scaled)
#The following two lines for univariate usad
#normal = normal.loc[:, 0]
#attack = attack.loc[:, 0]
window_size = args.train_window #9 ---> for better reconstruction #12

windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
windows_normal.shape
y_windows_normal =normal.values[(np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None] + window_size)[:, 0]]

windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]
windows_attack.shape
y_windows_attack =attack.values[(np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None] + window_size)[:, 0]]

import torch.utils.data as data_utils

BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
hidden_size = args.hidden_size
#batch, wnd = windows_normal.shape
#windows_normal = windows_normal.reshape(batch, wnd, 1)
#y_windows_normal = y_windows_normal.reshape(batch, 1)
#batch, wnd = windows_attack.shape
#windows_attack = windows_attack.reshape(batch, wnd, 1)
#y_windows_attack = y_windows_attack.reshape(batch, 1)
print(windows_normal.shape)
print(y_windows_normal.shape)
#batch, window_len, n_channels = windows_normal.shape
n_channels = 1

w_size=windows_normal.shape[1]*windows_normal.shape[2] #12*51 = 612
z_size=int(windows_normal.shape[1]*hidden_size) # 12*100 = 1200

#ENCODER:
#612 --> 306
#306 --> 153
#153 --> 1200

windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
y_windows_normal_train = y_windows_normal[:int(np.floor(.8 *  y_windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]
y_windows_normal_val = y_windows_normal[int(np.floor(.8 *  y_windows_normal.shape[0])):int(np.floor(y_windows_normal.shape[0]))]

if model_type != "usad" and model_type != "linear_ae" and model_type != "lstm":
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],windows_normal_train.shape[1], windows_normal_train.shape[2]])) #windows_normal_train.shape[2]
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],windows_normal_train.shape[1], windows_normal_train.shape[2]]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],windows_normal_train.shape[1], windows_normal_train.shape[2]]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
elif model_type == "lstm":
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(windows_normal_train).float(), torch.from_numpy(y_windows_normal_train).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(windows_normal_val).float(), torch.from_numpy(y_windows_normal_val).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(windows_attack).float(), torch.from_numpy(y_windows_attack).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
else:
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
    ) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print("Define model")
print(w_size)
print(z_size)
#model = UsadModel(w_size, z_size)
#z_size = 32
model = LinearAE(w_size, z_size)
print(model)
#model.to(device)
model = model.to(device) #to_device(model,device)

history = training(N_EPOCHS,model,train_loader,val_loader, device)

checkpoint_dir = args.save_checkpoint_dir

#plot_history(history)
if model_type == "lstm_ae" or model_type == "conv_ae" or model_type == "vae" or model_type == "lstm":
    history_to_save = torch.stack(history).flatten().detach().cpu().numpy()
    np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/history_linear_ae_swat_multi_ordered_05_06.npy', history_to_save) #/content/checkpoints er prove su drive
    #np.save('/content/checkpoints/train_recos.npy', train_recos_to_save)
else:
    np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/history_linear_ae_swat_multi_ordered_05_06.npy', history)
#np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/history_linear_ae_swat_uni_40_hs_to_device_torch.npy', history)

if model_type != "lstm":
    torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                #'decoder2': model.decoder2.state_dict()
                }, checkpoint_dir)  #"/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/usad_model_odin.pth"
else:
    torch.save(model.state_dict(), checkpoint_dir)

results, forec=testing(model,test_loader, device)

windows_labels=[]
for i in range(len(labels)-window_size):
    windows_labels.append(list(np.int_(labels[i:i+window_size])))

y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]

y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                              results[-1].flatten().detach().cpu().numpy()])

threshold=ROC(y_test,y_pred)
y_pred_ = np.zeros(y_pred.shape[0])
y_pred_[y_pred >= threshold] = 1
print(roc_auc_score(y_test, y_pred_))
print(classification_report(y_test, y_pred_))
print("OTHER METHOD: ")
threshold = np.percentile(y_pred, 93)
y_pred_ = np.zeros(y_pred.shape[0])
y_pred_[y_pred >= threshold] = 1
print(classification_report(y_test, y_pred_))
print(roc_auc_score(y_test, y_pred_))