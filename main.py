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

import warnings
warnings.filterwarnings('ignore')

device = get_default_device()


if __name__ == "__main__":
    args = parser_file.parse_arguments()



    #### Open the dataset ####
    energy_df = pd.read_csv(r"/home/medoro/Unsupervised_Anomaly_Detection_thesis/data/train.csv")
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
    #Overlapping windows
    X_test, y_test = create_train_eval_sequences(test, train_window)
    #Non overlapping windows
    #X_test, y_test = create_test_sequences(test, train_window)

    BATCH_SIZE =  args.batch_size
    N_EPOCHS = args.epochs
    hidden_size = args.hidden_size

    w_size = X_train.shape[1] * X_train.shape[2]
    z_size = int(w_size * hidden_size) 
    w_size, z_size

    model_type = args.model_type

    if model_type == "conv_ae":
        train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().view(([X_train.shape[0], w_size, 1]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
        val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().view(([X_val.shape[0],w_size, 1]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_test).float().view(([X_test.shape[0],w_size, 1]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    else:
        train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train).float().view(([X_train.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
        val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val).float().view(([X_val.shape[0],w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_test).float().view(([X_test.shape[0],w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)




    # Create the model and send it on the gpu device
    model = UsadModel(w_size, z_size)
    model = to_device(model, device)
    print(model)

    if args.do_train:
        # Start training
        history = training(N_EPOCHS, model, train_loader, val_loader)
        plot_history(history)
        checkpoint_path = args.save_checkpoint_dir
        torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder1': model.decoder1.state_dict(),
                'decoder2': model.decoder2.state_dict()
                }, checkpoint_path+"/model_100epochs_univariate.pth") # the path should be changed

    if args.do_testing:
        ### TESTING ###
        # Recover checkpoint
        checkpoint_dir = args.checkpoint_dir
        checkpoint = torch.load(checkpoint_dir+"/model_50epochs_uni_conv.pth")

        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder1.load_state_dict(checkpoint['decoder1'])
        model.decoder2.load_state_dict(checkpoint['decoder2'])

        w1_non_overl, w2_non_overl = reconstruction(model, test_loader)
        if model_type == "conv_ae":
            w1 = [torch.reshape(w1_el, (w1_el.size()[0], w1_el.size()[1])) for w1_el in w1_non_overl]
            w2 = [torch.reshape(w2_el, (w2_el.size()[0], w2_el.size()[1])) for w2_el in w2_non_overl]

        # w1
        reshaped_w1 = [torch.flatten(w1_el) for w1_el in w1_non_overl]
        stacked = torch.stack(reshaped_w1[:-1]).flatten()
        stacked_array = stacked.cpu().numpy()
        last_array = reshaped_w1[-1].cpu().numpy()
        total = np.concatenate([stacked_array, last_array])

        # w2
        reshaped_w2 = [torch.flatten(w2_el) for w2_el in w2_non_overl]
        stacked2 = torch.stack(reshaped_w2[:-1]).flatten()
        stacked_array2 = stacked2.cpu().numpy()
        last_array2 = reshaped_w2[-1].cpu().numpy()
        total2 = np.concatenate([stacked_array2, last_array2])


