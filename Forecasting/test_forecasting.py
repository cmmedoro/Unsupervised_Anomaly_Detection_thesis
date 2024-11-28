from preprocessing_forecast import *
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, recall_score, f1_score, precision_score
from postprocessing_forecast import *
import plotly.graph_objects as go
import torch.utils.data as data_utils
import parser_file
import warnings
warnings.filterwarnings('ignore')
from utils_ae import ROC
from lstm import *

args = parser_file.parse_arguments()

model_type = args.model_type    

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
# 2) Add trigonometric features
dfs_dict = impute_missing_dates(imputed_df)
df1 = pd.concat(dfs_dict.values())
#lags = [1, -1]
#df1 = create_diff_lag_features(df1, lags)
# 3) Add trigonometric features
#df2 = add_trigonometric_features(df1)
#lags = [-1, 24, -24, 168, -168]
#df2 = create_diff_lag_features(df2, lags)
df2 = resampling_daily(df1)

# Split the dataset into train, validation and test
dfs_train, dfs_val, dfs_test = train_val_test_split(df2)
train = pd.concat(dfs_train.values())
val = pd.concat(dfs_val.values())
test = pd.concat(dfs_test.values())

if args.do_resid:
    # Residuals dataset (missing values and dates imputation already performed)
    residuals = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/residuals2.csv")
    #residuals = pd.read_csv("/content/drive/MyDrive/ADSP/Backup_tesi_Carla_sorry_bisogno_di_gpu/residuals.csv")
    residui_df = residuals[['timestamp', 'building_id', 'primary_use', 'anomaly', 'meter_reading', 'sea_level_pressure', 'is_holiday', 'resid', 'air_temperature']]
    dfs_train, dfs_val, dfs_test = train_val_test_split(residui_df)
    train = pd.concat(dfs_train.values())
    val = pd.concat(dfs_val.values())
    test = pd.concat(dfs_test.values())
print(train.columns)
if args.do_multivariate:
    residuals = pd.read_csv("/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/residuals2.csv")
    residui_df = residuals[['timestamp', 'building_id', 'primary_use', 'anomaly', 'meter_reading', 'sea_level_pressure', 'is_holiday', 'resid', 'air_temperature']]
    residui_df = add_trig_resid(residui_df)
    lags = [-1, 24, -24, 168, -168]
    residui_df = create_diff_lag_features(residui_df, lags)
    residui_df = add_rolling_feature(residui_df, 12)
    residui_df = add_rolling_feature(residui_df, 24)
    dfs_train, dfs_val, dfs_test = train_val_test_split(residui_df)
    train = pd.concat(dfs_train.values())
    val = pd.concat(dfs_val.values())
    test = pd.concat(dfs_test.values())

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
if args.multistep:
    fh = args.horizon
    X_train, y_train = create_multistep_sequences(train, train_window, fh)
    X_val, y_val = create_multistep_sequences(val, train_window, fh)
    X_test, y_test = create_multistep_sequences(test, train_window)

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
model = LstmModel(n_channels, 32, fh)
model = model.to(device) #to_device(model, device)
print(model)

checkpoint_dir = args.checkpoint_dir
checkpoint = torch.load(checkpoint_dir) #map_location = torch.device('cpu')
model.load_state_dict(checkpoint)

results, forecast = testing(model, test_loader, device)
# On validation set
results_v, forecast_v = testing(model, val_loader, device)

if args.do_reconstruction: 
    forecast_test = np.concatenate([torch.stack(forecast[:-1]).flatten().detach().cpu().numpy(), forecast[-1].flatten().detach().cpu().numpy()])
    forecast_val = np.concatenate([torch.stack(forecast_v[:-1]).flatten().detach().cpu().numpy(), forecast_v[-1].flatten().detach().cpu().numpy()])
    np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/forec_test_wo_init_40_28_05.npy', forecast_test)
    np.save('/nfs/home/medoro/Unsupervised_Anomaly_Detection_thesis/checkpoints/forecval_wo_init_40_28_05.npy', forecast_val)
    predicted_df_val = get_predicted_dataset(val, forecast_val, train_window)
    predicted_df_test = get_predicted_dataset(test, forecast_test, train_window)

    threshold_method = args.threshold
    percentile = args.percentile
    weight_overall = args.weights_overall

    print("METHOD: ", threshold_method)

    predicted_df_test = anomaly_detection(predicted_df_val, predicted_df_test, threshold_method, percentile, weight_overall)
    print(predicted_df_test.columns)
    # The following two lines need to be commented in the multivariate case
    predicted_df_test.index.names=['timestamp']
    predicted_df_test= predicted_df_test.reset_index()

    predicted_df_test['timestamp']=predicted_df_test['timestamp'].astype(str)

    predicted_df_test = pd.merge(predicted_df_test, df[['timestamp','building_id']], on=['timestamp','building_id'])

    print(classification_report(predicted_df_test['anomaly'], predicted_df_test['predicted_anomaly']))
    print(roc_auc_score(predicted_df_test['anomaly'], predicted_df_test['predicted_anomaly']))
    print("Risultati corretti con post-processing su valori mancanti")
    predicted_df_test = postprocessing_on_missing_values(predicted_df_test, predicted_df_test)
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

    print("STATISTICHE PER Primary_use")
    precisions_p = []
    recalls_p = []
    rocs_p = []
    f1s_p = []
    scores_p = {}
    for primary_use, gdf in predicted_df_test.groupby("primary_use"):
        prec = precision_score(gdf['anomaly'], gdf['predicted_anomaly'])
        rec = recall_score(gdf['anomaly'], gdf['predicted_anomaly'])
        roc = roc_auc_score(gdf['anomaly'], gdf['predicted_anomaly'])
        f1 = f1_score(gdf['anomaly'], gdf['predicted_anomaly'])
        precisions_p.append(prec)
        recalls_p.append(rec)
        rocs_p.append(roc)
        f1s_p.append(f1)
        #print("Building: ", building_id)
        #print("Precision: {:.4f}; Recall: {:.4f}; F1: {:.4f}; ROC: {:.4f}".format(prec, rec, f1, roc))
        scores_p[primary_use] = [prec, rec, roc, f1]
    print("Scores by primary_use")
    print(scores_p)


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


y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                            results[-1].flatten().detach().cpu().numpy()])

threshold=ROC(y_test,y_pred)
y_pred_ = np.zeros(y_pred.shape[0])
y_pred_[y_pred >= threshold] = 1
print(roc_auc_score(y_test, y_pred_))
print(classification_report(y_test, y_pred_))
print("OTHER METHOD: ")
threshold = np.percentile(y_pred, 70)
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

print("STATISTICHE MEDIE - PER BUILDING")
precisions = []
recalls = []
rocs = []
f1s = []
scores = {}
for building_id, gdf in predicted_df.groupby("building_id"):
    prec = precision_score(gdf['windowed_labels'], gdf['predictions'])
    rec = recall_score(gdf['windowed_labels'], gdf['predictions'])
    roc = roc_auc_score(gdf['windowed_labels'], gdf['predictions'])
    f1 = f1_score(gdf['windowed_labels'], gdf['predictions'])
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
        if  ro > highest_score[2] and f > highest_score[3]: #p> highest_score[0] and re > highest_score[1] and and f > highest_score[3]
            highest_score = score
            best_building = bid
print("Building {} has the highest scores {}".format(best_building, highest_score))
print("Building type: ", predicted_df[predicted_df.building_id == best_building].primary_use.unique())

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
        if  ro < lowest_score[2]: #p> lowest_score[0] and re > lowest_score[1] and and f > lowest_score[3]
            lowest_score = score
            worst_building = bid
print("Building {} has the lowest scores {}".format(worst_building, lowest_score))
print("Building type: ", predicted_df[predicted_df.building_id == worst_building].primary_use.unique())

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



