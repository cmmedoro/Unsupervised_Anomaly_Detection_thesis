import torch
import torch.nn as nn

#from utils_ae import *
#device = get_default_device()

class Encoder(nn.Module):
  def __init__(self, in_size, latent_size): 
    super().__init__()
    self.lstm = nn.LSTM(input_size=in_size, hidden_size=latent_size, num_layers=1, batch_first=True, dropout = 0.2
            # input and output tensors are provided as (batch, seq_len, feature(size))
        )
    self.dropout = nn.Dropout(0.2)
  def forward(self, w):
    #print("Input E: ", w.size())
    z, (h_n, c_n) = self.lstm(w)
    #print("Output E: ", h_n.size())
    h_n = self.dropout(h_n)
    return h_n
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size, train_window): 
    super().__init__()
    self.latent_size = latent_size
    self.window = train_window
    self.lstm = nn.LSTM(input_size=latent_size, hidden_size=latent_size, num_layers=1, batch_first=True, dropout = 0.2
            # input and output tensors are provided as (batch, seq_len, feature(size))
        )
    self.dropout = nn.Dropout(0.2)
    self.output_layer = nn.Linear(latent_size, out_size)
        
  def forward(self, z):
    batch = z.size()[1]
    #n_feats = z.size()[2]
    #print("Input D: ", z.size())
    #z = z.reshape((batch, n_feats))
    z = z.squeeze()
    #print("Reshaped input: ", z.size())
    #input = z.reshape((batch, self.latent_size))
    input = z.repeat(1, self.window)
    #print(input.size())
    input = input.reshape((batch, self.window, self.latent_size))
    #print(input.size())
    w, (h_n, c_n) = self.lstm(input)
    #print("Out D: ", w.size())
    w = self.dropout(w)
    out = self.output_layer(w)
    #print("Output D: ", out.size())
    return out
    
class LstmAE(nn.Module):
  def __init__(self, input_dim, latent_size, train_window): 
    super().__init__()
    self.encoder = Encoder(input_dim, latent_size)
    self.decoder = Decoder(latent_size, input_dim, train_window)
  
  def training_step(self, batch, criterion, n):
    z = self.encoder(batch)
    w = self.decoder(z)
    loss = criterion(w, batch)#torch.mean((batch-w)**2) #loss = mse
    return loss

  def validation_step(self, batch, criterion, n):
    with torch.no_grad():
        z = self.encoder(batch)
        w = self.decoder(z)
        loss = criterion(w, batch)#torch.mean((batch-w)**2) #loss = mse
    return loss
    
  def epoch_end(self, epoch, result, result_train):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, result_train, result))
    
def evaluate(model, val_loader, criterion, device, n):
    batch_loss = []
    for [batch] in val_loader:
       batch = batch.to(device) # to_device(batch, device)
       loss = model.validation_step(batch, criterion, n) 
       batch_loss.append(loss)

    epoch_loss = torch.stack(batch_loss).mean()
    return epoch_loss


def training(epochs, model, train_loader, val_loader, device, opt_func=torch.optim.Adam): 
    history = []
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()))
    #criterion = nn.MSELoss().to(device) #nn.KLDivLoss(reduction="batchmean").to(device) #nn.MSELoss().to(device)
    criterion = nn.L1Loss().to(device)
    for epoch in range(epochs):
        train_loss = []
        for [batch] in train_loader:
            batch = batch.to(device) #to_device(batch,device)
            optimizer.zero_grad()

            loss = model.training_step(batch, criterion, epoch+1)
            train_loss.append(loss)
            loss.backward()
            optimizer.step()
            #optimizer.zero_grad()
        result_train = torch.stack(train_loss).mean()    
            
        result= evaluate(model, val_loader, criterion, device, epoch+1)
        #result_train = evaluate(model, train_loader, criterion, device, epoch+1)
        model.epoch_end(epoch, result, result_train)
        #model.epoch_end(epoch, result)
        history.append((result_train, result))
    return history 
    
def testing(model, test_loader, device):
    results=[]
    reconstruction = []
    criterion = nn.MSELoss().to(device) #nn.KLDivLoss(reduction="batchmean").to(device)
    with torch.no_grad():
        for [batch] in test_loader: 
            batch = batch.to(device) #to_device(batch,device)
            w=model.decoder(model.encoder(batch))
            # Need to squeeze the batch and reconstruction to compute correctly the anomaly score
            # This because the input and the reconstruction are 3-D tensors, so we need to turn them into 2-D
            batch_s = batch.reshape(-1, batch.size()[1] * batch.size()[2])
            w_s = w.reshape(-1, w.size()[1] * w.size()[2])
            #print(batch_s.size())
            #print(w_s.size())
            #results.append(criterion(w, batch))
            results.append(torch.mean((batch_s-w_s)**2,axis=1))
            reconstruction.append(w)
    #print(len(results), results[0].size())
    #print(len(reconstruction), reconstruction[0].size())
    return results, reconstruction
