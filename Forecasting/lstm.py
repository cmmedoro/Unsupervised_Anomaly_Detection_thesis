import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

from utils_ae import *
device = get_default_device()


class LstmModel(nn.Module):
  def __init__(self, in_size, latent_size): 
    super().__init__()
    """
    in_size: number of features in input
    latent_size: size of the latent space of the lstm
    Ex. in_size = 5, latent_size = 50
    """
    self.lstm = nn.LSTM(input_size=in_size, hidden_size=latent_size, num_layers=1, batch_first=True, dropout = 0.2
            # input and output tensors are provided as (batch, seq_len, feature(size))
        )
    self.dropout = nn.Dropout(0.2)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(latent_size, 1)

    # Initialize weights
    self.init_weights()

  def init_weights(self):
    for name, param in self.named_parameters():
      if 'weight' in name:
        nn.init.xavier_uniform_(param)
      elif 'bias' in name:
        nn.init.constant_(param, 0)
    
  def forward(self, w):
    #print("Input: ", w.size())
    z, (h_n, c_n) = self.lstm(w)
    #print(z[:,-1, :].size())
    forecast = z[:, -1, :]
    forecast = self.relu(forecast)
    forecast = self.dropout(forecast)
    output = self.fc(forecast)
    #print("Output 3: ", output.size())
    return output
  

def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), eps = 1e-07)
    criterion = nn.MSELoss().to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for X_batch, y_batch in train_loader:
          X_batch = to_device(X_batch,device)
          y_batch = to_device(y_batch, device)

          z = model(X_batch)
          loss = criterion(z.squeeze(), y_batch)
          train_loss.append(loss)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        result_train = torch.stack(train_loss).mean()

        model.eval()
        batch_loss = []
        for X_batch, y_batch in val_loader:
          X_batch = to_device(X_batch, device)
          y_batch = to_device(y_batch, device)
          with torch.no_grad():
            z = model(X_batch)
            loss = criterion(z, y_batch)
          #loss = model.validation_step(X_batch, y_batch, criterion, n)
          batch_loss.append(loss)

        result = torch.stack(batch_loss).mean()

        #result_train = evaluate(model, train_loader, criterion, epoch+1)
        #model.epoch_end(epoch, result, result_train)
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, result_train, result))


        history.append(result_train)
    return history
    
def testing(model, test_loader):
    results=[]
    forecast = []
    criterion = nn.MSELoss().to(device) #nn.KLDivLoss(reduction="batchmean").to(device)
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch=to_device(X_batch,device)
            y_batch = to_device(y_batch, device)
            w=model(X_batch)
            #results.append(criterion(w.squeeze(), y_batch))
            results.append(torch.mean((y_batch.unsqueeze(1)-w)**2,axis=1))
            forecast.append(w)
    return results, forecast
