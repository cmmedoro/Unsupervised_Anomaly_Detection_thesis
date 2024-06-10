from preprocessing import *
import preprocessing as prp
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from postprocessing import *
import plotly.graph_objects as go
import torch.utils.data as data_utils
import parser_file
from utils_ae import ROC
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

#device = get_default_device()

if torch.cuda.is_available():
    device =  torch.device('cuda')
else:
    device = torch.device('cpu')

#### Open the dataset ####
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
df2 = add_trigonometric_features(df1) #

# Split the dataset into train, validation and test
dfs_train, dfs_val, dfs_test = train_val_test_split(df2)
train = pd.concat(dfs_train.values())
val = pd.concat(dfs_val.values())
test = pd.concat(dfs_test.values())

if args.do_resid:
    # Residuals dataset (missing values and dates imputation already performed)
    residuals = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/residuals2.csv")
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
    residui_df = add_trig_resid(residui_df)
    dfs_train, dfs_val, dfs_test = train_val_test_split(residui_df)
    train = pd.concat(dfs_train.values())
    val = pd.concat(dfs_val.values())
    test = pd.concat(dfs_test.values())

### TRAINING THE MODEL ###
# For training we are going to create an input dataset consisting of overlapping windows of 72 measurements (3 days)
train_window = args.train_window
#X_train, y_train = create_train_eval_sequences(train, train_window)

if args.do_multivariate:
    X_train, y_train = create_multivariate_train_eval_sequences(train, train_window)
    X_val, y_val = create_multivariate_train_eval_sequences(val, train_window)
else:
    #X_train, y_train = create_train_eval_sequences(train, train_window)
    #X_val, y_val = create_train_eval_sequences(val, train_window)
    X_train, y_train = create_sequences(train, train_window, 1)
    X_val, y_val = create_sequences(val, train_window, 1)

if args.do_test and not args.do_multivariate:
    # Overlapping windows
    print("UNIV")
    X_test, y_test = create_train_eval_sequences(test, train_window)
    X_val, y_val = create_train_eval_sequences(val, train_window)
elif args.do_test and args.do_multivariate:
    print("MULTI")
    X_test, y_test = create_multivariate_train_eval_sequences(test, train_window)
    X_val, y_val = create_multivariate_train_eval_sequences(val, train_window)
elif args.do_reconstruction and args.do_multivariate:
    X_test, y_test = create_multivariate_test_sequences(test, train_window)
    X_val, y_val = create_multivariate_test_sequences(val, train_window)
else:
    # Non-overlapping windows
    X_test, y_test = create_test_sequences(test, train_window)
    X_val, y_val = create_test_sequences(val, train_window)
    print(X_test.shape, y_test.shape)

print("X_train: ", X_train.shape)
print("X_val: ", X_val.shape)
print("X_test: ", X_test.shape)
BATCH_SIZE =  args.batch_size
N_EPOCHS = args.epochs
hidden_size = args.hidden_size

# batch_size, window_length, n_features = X_train.shape
batch, window_len, n_channels = X_train.shape

w_size = X_train.shape[1] * X_train.shape[2]
z_size = int(w_size * hidden_size) 
w_size, z_size



if model_type == "conv_ae" or model_type == "lstm_ae" or model_type == "usad_conv" or model_type == "usad_lstm" or model_type == "vae":
    #train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().view(([X_train.shape[0], w_size, 1]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    #Credo di dover cambiare X_val.shape[0], w_size, X_val.shape[2] con X_val.shape[0], X_val.shape[1], X_val.shape[2]
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().view(([X_val.shape[0], X_val.shape[1], X_val.shape[2]]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_test).float().view(([X_test.shape[0],X_test.shape[1], X_test.shape[2]]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
else:
    #train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().view(([X_train.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().reshape(([X_val.shape[0],w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_test).float().reshape(([X_test.shape[0],w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

if model_type == "lstm_ae" or model_type == "conv_ae" or model_type == "vae":
    z_size = 32
# Create the model and send it on the gpu device
if model_type == "lstm_ae":
    model = LstmAE(n_channels, z_size, train_window)
elif model_type == "conv_ae":
    model = ConvAE(n_channels, z_size)
elif model_type == "linear_ae":
    model = LinearAE(w_size, z_size)
elif model_type == "vae":
    model = LstmVAE(n_channels, z_size, train_window)
else:
    model = UsadModel(w_size, z_size)
model = model.to(device) #to_device(model, device)
print(model)

if args.do_reconstruction:
    ### RECONSTRUCTING ###
    # Recover checkpoint
    checkpoint_dir = args.checkpoint_dir
    checkpoint = torch.load(checkpoint_dir)

    if model_type.startswith("usad"):
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder1.load_state_dict(checkpoint['decoder1'])
        model.decoder2.load_state_dict(checkpoint['decoder2'])
    else: 
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])

    if model_type == "vae":
        results, w, mu, logvar = testing(model, test_loader, device)
    else:
        results, w = testing(model,test_loader, device)
    print(len(w), w[0].size())
    # Reconstruction of validation set
    if model_type == "vae":
        results_v, w_v, mu_v, logvar_v = testing(model, val_loader, device)
    else:
        results_v, w_v = testing(model,val_loader, device)


    res_dir = args.res_dir

    reconstruction_test = np.concatenate([torch.stack(w[:-1]).flatten().detach().cpu().numpy(), w[-1].flatten().detach().cpu().numpy()])
    reconstruction_val = np.concatenate([torch.stack(w_v[:-1]).flatten().detach().cpu().numpy(), w_v[-1].flatten().detach().cpu().numpy()])
    
    predicted_df_val = get_predicted_dataset(val, reconstruction_val)
    predicted_df_test = get_predicted_dataset(test, reconstruction_test)

    threshold_method = args.threshold
    percentile = args.percentile
    weight_overall = args.weights_overall

    predicted_df_test = anomaly_detection(predicted_df_val, predicted_df_test, threshold_method, percentile, weight_overall)
    predicted_df_test.index.names=['timestamp']
    predicted_df_test= predicted_df_test.reset_index()
    predicted_df_test['timestamp']=predicted_df_test['timestamp'].astype(str)

    predicted_df_test = pd.merge(predicted_df_test, df[['timestamp','building_id']], on=['timestamp','building_id'])

    print(classification_report(predicted_df_test['anomaly'], predicted_df_test['predicted_anomaly']))
    print(roc_auc_score(predicted_df_test['anomaly'], predicted_df_test['predicted_anomaly']))

    print("STATISTICHE MEDIE - PER BUILDING")
    precisions = []
    recalls = []
    rocs = []
    f1s = []
    scores = {}
    for building_id, gdf in predicted_df_test.groupby("building_id"):
        prec = precision_score(gdf['anomaly'], gdf['predicted_anomaly'])
        rec = recall_score(gdf['anomaly'], gdf['predicted_anomaly'])
        roc = roc_auc_score(gdf['anomaly'], gdf['predicted_anomaly'])
        f1 = f1_score(gdf['anomaly'], gdf['predicted_anomaly'])
        precisions.append(prec)
        recalls.append(rec)
        rocs.append(roc)
        f1s.append(f1)
        #print("Building: ", building_id)
        #print("Precision: {:.4f}; Recall: {:.4f}; F1: {:.4f}; ROC: {:.4f}".format(prec, rec, f1, roc))
        scores[building_id] = [prec, rec, roc, f1]
    print("Average scores by building")
    print(np.mean(precisions))
    print(np.mean(recalls))
    print(np.mean(rocs))
    print(np.mean(f1s))

    

    print("Highest score and corresponding building and building type")
    highest_score = []
    best_building = None
    for bid, score in scores.items():
        #print("Building: ", bid)
        #print("Precision: {:.4f}; Recall: {:.4f}; F1: {:.4f}; ROC: {:.4f}".format(score[0], score[1], score[3], score[2]))
        if len(highest_score) == 0:
            highest_score = score
            best_building = bid
        else:
            p, re, ro, f = score
            if  ro > highest_score[2] : #p> highest_score[0] and re > highest_score[1] and and f > highest_score[3]
                highest_score = score
                best_building = bid
    print("Building {} has the highest scores {}".format(best_building, highest_score))
    print("Building type: ", predicted_df_test[predicted_df_test.building_id == best_building].primary_use.unique())

    print("Lowest score and corresponding building and building type")
    lowest_score = []
    worst_building = None
    for bid, score in scores.items():
        #print("Building: ", bid)
        #print("Precision: {:.4f}; Recall: {:.4f}; F1: {:.4f}; ROC: {:.4f}".format(score[0], score[1], score[3], score[2]))
        if len(lowest_score) == 0:
            lowest_score = score
            worst_building = bid
        else:
            p, re, ro, f = score
            if  ro < lowest_score[2] : #p> lowest_score[0] and re > lowest_score[1] and and f > lowest_score[3]
                lowest_score = score
                worst_building = bid
    print("Building {} has the lowest scores {}".format(worst_building, lowest_score))
    print("Building type: ", predicted_df_test[predicted_df_test.building_id == worst_building].primary_use.unique())

    print("EVALUATION: point anomalies VS Contextual")
    predicted_df_test['anomaly_outlier'] = [1 if (el['anomaly']==1 and el['outliers']==1) else 0 for _ , el in predicted_df_test.iterrows()]
    predicted_df_test['predicted_anomaly_outlier'] = [1 if (el['predicted_anomaly']==1 and el['outliers']==1) else 0 for _ , el in predicted_df_test.iterrows()]
    predicted_df_test['anomaly_not_outlier'] = predicted_df_test['anomaly']-predicted_df_test['anomaly_outlier']
    predicted_df_test['predicted_anomaly_not_outlier'] = predicted_df_test['predicted_anomaly']-predicted_df_test['predicted_anomaly_outlier']
    print("Results for Outliers: ")
    print(classification_report(predicted_df_test['anomaly_outlier'], predicted_df_test['predicted_anomaly_outlier']))
    print(roc_auc_score(predicted_df_test['anomaly_outlier'], predicted_df_test['predicted_anomaly_outlier']))
    print("Results for contextual anomalies: ")
    print(classification_report(predicted_df_test['anomaly_not_outlier'], predicted_df_test['predicted_anomaly_not_outlier']))
    print(roc_auc_score(predicted_df_test['anomaly_not_outlier'], predicted_df_test['predicted_anomaly_not_outlier']))



elif args.do_test:
    ### TESTING ###
    # Recover checkpoint
    checkpoint_dir = args.checkpoint_dir
    checkpoint = torch.load(checkpoint_dir)

    if model_type.startswith("usad"):
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder1.load_state_dict(checkpoint['decoder1'])
        model.decoder2.load_state_dict(checkpoint['decoder2'])
    else: 
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])

    #results, w = testing(model,test_loader)
    #if not model_type.startswith("usad"):
     #   results, w = testing(model, test_loader)
    #else:
     #   results = testing(model, test_loader)
    if model_type == "vae":
        results, w, mu, logvar = testing(model, test_loader, device)
    elif model_type == "usad":
        results = testing(model, test_loader, device)
    else:
        results, w = testing(model,test_loader, device)


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

    y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(), results[-1].flatten().detach().cpu().numpy()])
    
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

    print("STATISTICHE MEDIE - PER BUILDING")



