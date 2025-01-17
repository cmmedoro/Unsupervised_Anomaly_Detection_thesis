from ad_production.preprocessing import *
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

if model_type == "linear_ae":
    from linear_ae import *
    from utils_ae import *
elif model_type == "conv_ae":
    from convolutional_ae import *
    from utils_ae import *
elif model_type == "lstm_ae":
    from lstm_ae import *
    from utils_ae import *

#device = get_default_device()

if torch.cuda.is_available():
    device =  torch.device('cuda')
else:
    device = torch.device('cpu')

#### Open the dataset ####
# Original dataset
production_df = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/ad_production_dc.csv")
production_df['datetime'] = pd.to_datetime(production_df.datetime)
# Resampling of dates
production_df = impute_missing_dates(production_df)
# Select some columns from the original dataset
production_df1 = production_df[['generation_kwh']]

### PREPROCESSING ###
# Split the dataset into train, validation and test
dfs_train, dfs_val, dfs_test = split(production_df1)
test = dfs_test.reset_index(drop = True)

scaler = MinMaxScaler(feature_range = (0,1))
X_train = scaler.fit_transform(dfs_train)
X_val = scaler.transform(dfs_val)
X_test = scaler.transform(test)
### TRAINING THE MODEL ###
# For training we are going to create an input dataset consisting of overlapping windows of 72 measurements (3 days)
#### CAMBIA LE FEATURES DA TENERE IN CASO MULTIVARIATO
train_window = args.train_window
X_t = create_sequences(X_train, train_window, train_window)
X_v = create_sequences(X_val, train_window, train_window)
X_te = create_sequences(X_test, train_window, train_window)


BATCH_SIZE =  args.batch_size
N_EPOCHS = args.epochs
hidden_size = args.hidden_size

# batch_size, window_length, n_features = X_train.shape
batch, window_len, n_channels = X_train.shape

w_size = X_train.shape[1] * X_train.shape[2]
z_size = int(w_size * hidden_size) 

if model_type == "conv_ae" or model_type == "lstm_ae" :
    #Credo di dover cambiare X_train.shape[0], w_size, X_train.shape[2] con X_train.shape[0], X_train.shape[1], X_train.shape[2]
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_v).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_te).float()) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
   
elif model_type == "linear_ae" and args.do_multivariate:
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_v).float().reshape(([X_v.shape[0], w_size])), torch.from_numpy(X_v).float().reshape(([X_v.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_te).float().reshape(([X_te.shape[0], w_size])), torch.from_numpy(X_te).float().reshape(([X_te.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
else:
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_v).float().reshape(([X_v.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_te).float().reshape(([X_te.shape[0],w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

#if model_type == "lstm_ae" or model_type == "conv_ae":
 #   z_size = 32
# Create the model and send it on the gpu device
if model_type == "lstm_ae":
    model = LstmAE(n_channels, z_size, train_window)
elif model_type == "conv_ae":
    model = ConvAE(n_channels, z_size) #n_channels
elif model_type == "linear_ae":
    model = LinearAE(w_size, z_size)
model = model.to(device) #to_device(model, device)
print(model)

if args.do_reconstruction:
    ### RECONSTRUCTING ###
    # Recover checkpoint
    checkpoint_dir = args.checkpoint_dir
    checkpoint = torch.load(checkpoint_dir)

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])


    results, w = testing(model, test_loader, device)
    print(len(w), w[0].size())
    # Reconstruction of training set
    results_v, w_v = testing(model, val_loader, device)


    res_dir = args.res_dir

    reconstruction_test = w[0].flatten().detach().cpu().numpy()
    reconstruction_val = w_v[0].flatten().detach().cpu().numpy()
    print(len(reconstruction_test))
    print(len(reconstruction_val))
    dim_val = X_v.shape[0] * X_v.shape[1]
    dim_test = X_te.shape[0] * X_te.shape[1]
    predicted_df_val = get_predicted_dataset(pd.DataFrame(X_val[:dim_val], columns = ['generation_kwh']), reconstruction_val)
    predicted_df_test = get_predicted_dataset(pd.DataFrame(X_test[:dim_test], columns=['generation_kwh']), reconstruction_test)

    threshold_method = args.threshold
    percentile = args.percentile
    weight_overall = args.weights_overall

    print("Method: ", threshold_method)

    predicted_df_test = anomaly_detection(predicted_df_val, predicted_df_test, threshold_method, percentile, weight_overall)

    ### Parallel with synthetically generated anomalies on the data
    ub_an = int(X_test.shape[0] * 0.03)
    indices_to_zero = [258, 259, 260, 261, 262, 263, 264, 265, 266, 267]
    synthetic_df = synthetize_anomalies(X_test, ub_an, indices_to_zero)
    synthetic_df = synthetic_df.reset_index(drop = True)
    synth = synthetic_df[['generation_kwh']]
    X_synth = scaler.transform(synth)
    synth_seq = create_sequences(X_synth, train_window, train_window)

    synth_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(synth_seq).float().view(([synth_seq.shape[0], w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    if model_type == "conv_ae" or model_type == "lstm_ae":
        synth_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(synth_seq).float()) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    elif model_type == "linear_ae":
        synth_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(synth_seq).float().view(([synth_seq.shape[0], w_size])), torch.from_numpy(synth_seq).float().view(([synth_seq.shape[0], w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    else:
        synth_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(synth_seq).float().view(([synth_seq.shape[0], w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    res_s, w_s = testing(model, synth_loader, device)
    r_s = w_s[0].flatten().detach().cpu().numpy()
    dim_s = synth_seq.shape[0] * synth_seq.shape[1]
    df_s = get_predicted_dataset(pd.DataFrame(X_synth[:dim_s], columns = ['generation_kwh']), r_s)

    preds_s = anomaly_detection(predicted_df_val, df_s, threshold_method, percentile, weight_overall)

    tc = synthetic_df[['synthetic_anomaly']]
    ss = pd.concat([preds_s, tc[:dim_s]], axis = 1)

    print(classification_report(ss.synthetic_anomaly, ss.predicted_anomaly))
    anomalized_df = ss[ss.synthetic_anomaly == 1]
    print(classification_report(anomalized_df.synthetic_anomaly, anomalized_df.predicted_anomaly))
    """
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
    """



elif args.do_test:
    ### TESTING ###
    # Recover checkpoint
    checkpoint_dir = args.checkpoint_dir
    checkpoint = torch.load(checkpoint_dir)

    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])

    #results, w = testing(model,test_loader)
    #if not model_type.startswith("usad"):
     #   results, w = testing(model, test_loader)
    #else:
     #   results = testing(model, test_loader)
    results, w = testing(model,test_loader, device)


    # Qui va ad ottenere le label per ogni finestra
    # Input modello è una lista di array, ognuno corrispondente a una sliding window con stride = 1 sui dati originali
    # Quindi dobbiamo applicare la sliding window anche sulle label
    windows_labels=[]
    for b_id, gdf in test.groupby('building_id'):
        labels = gdf.anomaly.values
        for i in range(len(labels)-train_window):
            windows_labels.append(list(np.int_(labels[i:i+train_window])))
    #windows_labels

    scaler = MinMaxScaler(feature_range = (0,1))
    dfs_dict_1 = {}
    for building_id, gdf in test.groupby("building_id"):
        gdf[['meter_reading', 'sea_level_pressure']]=scaler.fit_transform(gdf[['meter_reading', 'sea_level_pressure']])
        upper_outlier_value=(np.percentile(gdf['meter_reading'].values, 75)) + 1.5 *((np.percentile(gdf['meter_reading'].values, 75))-(np.percentile(gdf['meter_reading'].values, 25)))
        lower_outlier_value = (np.percentile(gdf['meter_reading'].values, 25)) - 1.5 *((np.percentile(gdf['meter_reading'].values, 75))-(np.percentile(gdf['meter_reading'].values, 25)))
        gdf['outliers'] = [1 if (el<lower_outlier_value or el>upper_outlier_value) else 0 for el in gdf['meter_reading'].values]
        dfs_dict_1[building_id] = gdf
    new_test = pd.concat(dfs_dict_1.values())

    # Obtain the outliers windows labels
    outlier_labels=[]
    for b_id, gdf in new_test.groupby('building_id'):
        labels = gdf.outliers.values
        for i in range(len(labels)-train_window):
            outlier_labels.append(list(np.int_(labels[i:i+train_window])))


    y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]
    y_outlier = [1.0 if (np.sum(window) > 0) else 0 for window in outlier_labels ]

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


    dfs_dict_1 = {}
    for building_id, gdf in test.groupby("building_id"):
        gdf[['meter_reading']]=scaler.fit_transform(gdf[['meter_reading']])
        dfs_dict_1[building_id] = gdf[train_window:]
    predicted_df = pd.concat(dfs_dict_1.values())
    predicted_df['anomaly_score'] = y_pred
    predicted_df['predictions'] = y_pred_
    predicted_df['windowed_labels'] = y_test
    predicted_df['outliers_labels'] = y_outlier

    
    print("EVALUATION: point anomalies VS Contextual")
    predicted_df['anomaly_outlier'] = [1 if (el['windowed_labels']==1 and el['outliers_labels']==1) else 0 for _ , el in predicted_df.iterrows()]
    predicted_df['predicted_anomaly_outlier'] = [1 if (el['predictions']==1 and el['outliers_labels']==1) else 0 for _ , el in predicted_df.iterrows()]
    predicted_df['anomaly_not_outlier'] = predicted_df['windowed_labels']-predicted_df['anomaly_outlier']
    predicted_df['predicted_anomaly_not_outlier'] = predicted_df['predictions']-predicted_df['predicted_anomaly_outlier']
    print("Results for Outliers: ")
    print(classification_report(predicted_df['anomaly_outlier'], predicted_df['predicted_anomaly_outlier']))
    print(roc_auc_score(predicted_df['anomaly_outlier'], predicted_df['predicted_anomaly_outlier']))
    print("Results for contextual anomalies: ")
    print(classification_report(predicted_df['anomaly_not_outlier'], predicted_df['predicted_anomaly_not_outlier']))
    print(roc_auc_score(predicted_df['anomaly_not_outlier'], predicted_df['predicted_anomaly_not_outlier']))


