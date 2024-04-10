from preprocessing import *
import preprocessing as prp
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from usad import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from postprocessing import *
import plotly.graph_objects as go
import torch.utils.data as data_utils
#from usad_conv import *
import parser_file


device = get_default_device()

args = parser_file.parse_arguments()


#### Open the dataset ####
energy_df = pd.read_csv(r"/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/train.csv")
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

w_size = X_train.shape[1] * X_train.shape[2]
z_size = int(w_size * hidden_size) 
w_size, z_size

model_type = args.model_type

if model_type == "conv_ae":
    #train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().view(([X_train.shape[0], w_size, 1]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    #val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().view(([X_val.shape[0],w_size, 1]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_test).float().view(([X_test.shape[0],w_size, 1]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
else:
    #train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().view(([X_train.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    #val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().view(([X_val.shape[0],w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_test).float().view(([X_test.shape[0],w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Create the model and send it on the gpu device
model = UsadModel(w_size, z_size)
model = to_device(model, device)
print(model)

if args.do_reconstruction:
    ### RECONSTRUCTING ###
    # Recover checkpoint
    checkpoint_dir = args.checkpoint_dir
    checkpoint = torch.load(checkpoint_dir)

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder1.load_state_dict(checkpoint['decoder1'])
    model.decoder2.load_state_dict(checkpoint['decoder2'])

    w1_non_overl, w2_non_overl = reconstruction(model, test_loader)

    res_dir = args.res_dir

    torch.save({
        'w1': w1_non_overl,
        'w2': w2_non_overl
    }, res_dir)

    #In realtà forse la parte dopo può essere tutta messa in un notebook: basta salvare gli output della ricostruzione
    #Perchè il fatto è che semplicemente non si possono usare i notebook per fare training del modello, ma da qui in poi
    #si tratta di visualizzare i risultati
    if model_type == "conv_ae":
        w1 = [torch.reshape(w1_el, (w1_el.size()[0], w1_el.size()[1])) for w1_el in w1_non_overl]
        w2 = [torch.reshape(w2_el, (w2_el.size()[0], w2_el.size()[1])) for w2_el in w2_non_overl]

    # w1
    total_w1 = get_wi_reconstructed(w1) #OR: w1_non_overl
    
    # w2
    total_w2 = get_wi_reconstructed(w2) #OR: w2_non_overl

    # ANOMALY DETECTION
    pred_test = get_anomaly_dataset(test, total_w1, total_w2)

    pred_test['relative_loss1'] = np.abs((pred_test['reconstruction1']-pred_test['meter_reading'])/pred_test['reconstruction1'])
    pred_test['relative_loss2'] = np.abs((pred_test['reconstruction2']-pred_test['meter_reading'])/pred_test['reconstruction2'])

    # Thresholds
    pred_test = define_threshold(pred_test, 1)
    pred_test = define_threshold(pred_test, 2)

    pred_test.index.names=['timestamp']
    pred_test= pred_test.reset_index()

    pred_test = pd.merge(pred_test, df[['timestamp','building_id']], on=['timestamp','building_id'])

    
    print(classification_report(pred_test['anomaly'], pred_test['predicted_anomaly']))
    print(classification_report(pred_test['anomaly'], pred_test['predicted_anomaly2']))
    print(roc_auc_score(pred_test['anomaly'], pred_test['predicted_anomaly']))
    print(roc_auc_score(pred_test['anomaly'], pred_test['predicted_anomaly2']))
elif args.do_test:
    ### TESTING ###
    # Recover checkpoint
    checkpoint_dir = args.checkpoint_dir
    checkpoint = torch.load(checkpoint_dir+"/model_50epochs_uni_conv.pth")

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder1.load_state_dict(checkpoint['decoder1'])
    model.decoder2.load_state_dict(checkpoint['decoder2'])

    results = testing(model,test_loader)

    pred_test = get_overl_anomaly(train_window, test)

    res_list = []
    for el in results:
        for el2 in el:
            res_list.append(el2.cpu().item())

    pred_test['anomaly_score'] = res_list

    pred_test = define_overl_threshold(pred_test, 90)

    pred_test.index.names=['timestamp']
    pred_test= pred_test.reset_index()

    pred_test = pd.merge(pred_test, df[['timestamp','building_id']], on=['timestamp','building_id'])

    print(classification_report(pred_test.anomaly, pred_test.predicted_anomaly))
    roc_auc_score(pred_test['anomaly'], pred_test['predicted_anomaly'])




