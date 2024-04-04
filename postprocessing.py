import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import torch

# Define a function to revert the sliding window application
def reconstruction_windows(timeseries):
  """
  This function takes as input the windows regarding a single time series in the original dataset.
  Input: timeseries.shape = (8713, 72), where 72 = window_size, 8713 = number_of_windows = 24*366 - window_size + 1
  Output: media.shape = (24*366, 1) ---> meter_reading column reconstructed, by averaging the reconstructions for the same points
  """
  df_x_train = pd.DataFrame(np.squeeze(timeseries))
  transposed_df = df_x_train.T
  nan_rows = np.full((8784-transposed_df.shape[0],transposed_df.shape[1]),np.nan)
  rows = pd.DataFrame(nan_rows)
  transposed_df = pd.concat([transposed_df, rows], ignore_index = True) #transposed_df.append(rows, ignore_index = True)
  for i in range(0, transposed_df.shape[1]):
    col = transposed_df.iloc[:, i].shift(i)
    transposed_df.iloc[:, i] = col
  retransposed = transposed_df.T
  media = np.nanmean(retransposed, axis = 0)
  return media

def apply_reconstruction(dataframe, n_timeseries):
  # Take the entire dataframe
  squeezed_df = np.squeeze(dataframe)
  # Reshape it: ex. train (162, 8713, 72) ---> 162 = number of timeseries in the dataframe
  reshaped = np.reshape(squeezed_df, (n_timeseries, 8712, 72)) #8713
  reconstruction = []
  i = 0
  for timeseries in reshaped:
    # Reconstruct separately each timeseries
    print(i)
    average_reconstruction = reconstruction_windows(timeseries)
    reconstruction.append(average_reconstruction)
    i = i+1
  final_reconstruction = np.squeeze(reconstruction)
  fr = np.reshape(final_reconstruction, (n_timeseries*8784, 1))
  return fr
  
def padding_w(w, batch_size):
  # This function needs to be used on the outputs of the decoders: the last "batch" is not going to be full of the batch_size elements 
  # characterizing a single batch, therefore a proper padding needs to be ensured. Moreover, with this function, we also transform
  # the padded output into a suitable format to perform the following operations to obtain the reconstructed input
  last = w[-1]
  padded_last = F.pad(last, (0, 0, 0, batch_size-last.size()[0]))
  new_lista = w[:-1]
  new_lista.append(padded_last)
  res_w = torch.cat(new_lista, dim=0)
  index = batch_size-last.size()[0]
  original_rec = res_w[:-index]
  origin_rec = original_rec.detach().cpu().numpy()

  return origin_rec
