from preprocessing import *
import preprocessing as prp
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from USAD.usad import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from postprocessing import *
import plotly.graph_objects as go
#from usad_conv import *
import parser_file

import warnings
warnings.filterwarnings('ignore')

device = get_default_device()
#device = torch.device("cuda")

print(device)

args = parser_file.parse_arguments()

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

window_size = args.train_window #9 ---> for better reconstruction #12

windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
windows_normal.shape

windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]
windows_attack.shape

import torch.utils.data as data_utils

BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
hidden_size = args.hidden_size

w_size=windows_normal.shape[1]*windows_normal.shape[2] #12*51 = 612
z_size=int(windows_normal.shape[1]*hidden_size) # 12*100 = 1200

#ENCODER:
#612 --> 306
#306 --> 153
#153 --> 1200

windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

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
model = UsadModel(w_size, z_size)
#model.to(device)
model = to_device(model,device)

history = training(N_EPOCHS,model,train_loader,val_loader)

checkpoint_dir = args.save_checkpoint_dir

#plot_history(history)
np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/history_usad_odin.npy', history)

torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
            }, checkpoint_dir)  #"/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/usad_model_odin.pth"