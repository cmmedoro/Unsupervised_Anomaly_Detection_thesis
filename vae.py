import torch
import torch.nn as nn

from utils_ae import *
device = get_default_device()

class Encoder(nn.Module):
  def __init__(self, in_size, latent_size, latent_dim): 
    """
    in_size: number of features of the input dataset
    latent_size: number of cells of the lstm (for example, 32)
    latent_dim: dimension of the variational layer
    """
    super().__init__()
    self.lstm = nn.LSTM(input_size=in_size, hidden_size=latent_size, num_layers=1, batch_first=True, dropout = 0.2
            # input and output tensors are provided as (batch, seq_len, feature(size))
        )
    self.dropout = nn.Dropout(0.2)
  def forward(self, w):
    #print("Input E: ", w.size())
    z, (h_n, c_n) = self.lstm(w)
    #print("Z: ", z.size())
    #print("H_n: ", h_n.size())
    z = self.dropout(z)
    h_n = self.dropout(h_n)
    return h_n
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size, train_window): 
    super().__init__()
    """
    latent_size: input number of features
    out_size: number of values outputted by the linear layer (in theory, the linear layer outputs one point at a time)
    """
    self.latent_size = latent_size
    self.window = train_window
    self.lstm = nn.LSTM(input_size=latent_size, hidden_size=latent_size, num_layers=1, batch_first=True, dropout = 0.2
            # input and output tensors are provided as (batch, seq_len, feature(size))
        )
    self.dropout = nn.Dropout(0.2)
    self.output_layer = nn.Linear(latent_size, out_size)
        
  def forward(self, z, h):
    #print("Hidden state: ", h.size())
    #print("Z_reparametrized: ", z.size())
    batch = z.size()[0]
    #batch = z.size()[1]
    #n_feats = z.size()[2]
    #print("Input D: ", z.size())
    #z = z.reshape((batch, n_feats))
    input = z.repeat(1, self.window)
    #print("Repeated: ", input.size())
    input = input.reshape((batch, self.window, self.latent_size))
    #print("Reshaped: ", input.size())
    w, (h_n, c_n) = self.lstm(input)#, h)
    #print("Out D: ", w.size())
    w = self.dropout(w)
    out = self.output_layer(w)
    #print("Output D: ", out.size())
    return out
    
class LstmVAE(nn.Module):
  def __init__(self, input_dim, latent_size, train_window): 
    super().__init__()
    self.encoder = Encoder(input_dim, latent_size, latent_size)
    self.decoder = Decoder(latent_size, input_dim, train_window)
    self.mean = nn.Linear(latent_size, latent_size)
    self.log_var = nn.Linear(latent_size, latent_size)

  def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        noise = to_device(noise, device)

        z = mu + noise * std
        return z

  def regularization_loss(self, mu, logvar):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim = 1))
        return kld_loss
  
  def training_step(self, batch, criterion, n):
    h = self.encoder(batch)
    #batch_1 = h.size()[1]
    #n_feats = h.size()[2]
    #print("Input D: ", z.size())
    #h = h.reshape((batch_1, n_feats))
    h = h.squeeze()
    mu = self.mean(h)
    logvar = self.log_var(h)
    #print("MU:", mu.size())
    #print(logvar.size())
    z_hat = self.reparametrize(mu, logvar)
    #print("Z_reparametrized: ", z_hat.size())
    w = self.decoder(z_hat, (h, h))
    print("Output D: ", w.size())
    #print("Batch type: ", type(batch))
    print("Batch: ", batch.size())
    loss_1 = criterion(w, batch)
    loss_2 = self.regularization_loss(mu, logvar)
    #loss_3 = torch.mean(self.regularization_loss(mu, logvar), dim = 0)
    kld_weight = 0.015 
    loss = loss_1 + loss_2 * kld_weight
    #print(loss)
    #loss_final = torch.mean(loss, dim = 0)
    return loss, w

  def validation_step(self, batch, criterion, n):
    with torch.no_grad():
        h = self.encoder(batch) #, z_hat, mu, logvar
        h = h.squeeze()
        mu = self.mean(h)
        logvar = self.log_var(h)
        z_hat = self.reparametrize(mu, logvar)
        w = self.decoder(z_hat, (h, h))
        loss_1 = criterion(w, batch) # torch.mean((batch-w)**2)
        loss_2 = self.regularization_loss(mu, logvar)
        kld_weight = 0.015 
        loss = loss_1 + loss_2 * kld_weight#torch.mean((batch-w)**2) #loss = mse
    return loss
    
  def epoch_end(self, epoch, result, result_train):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, result_train, result))
    
def evaluate(model, val_loader, criterion, n):
    batch_loss = []
    for [batch] in val_loader:
       batch = to_device(batch, device)
       loss = model.validation_step(batch, criterion, n) 
       batch_loss.append(loss.mean())

    epoch_loss = torch.stack(batch_loss).mean()
    return epoch_loss


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam): 
    history = []
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()))
    # Setup loss function
    criterion = nn.MSELoss().to(device)
    train_recos = []
    for epoch in range(epochs):
        model.train()
        tr_rec = []
        for [batch] in train_loader:
            batch=to_device(batch,device)
            optimizer.zero_grad()

            loss, train_reco = model.training_step(batch, criterion, epoch+1)
            #print("Loss: ", loss.mean())
            loss.mean().backward() #loss.mean().backward()
            optimizer.step()
            
            tr_rec.append(train_reco)

        train_recos.append(tr_rec) 
        model.eval()
        result= evaluate(model, val_loader, criterion, epoch+1)
        result_train = evaluate(model, train_loader, criterion, epoch+1)
        model.epoch_end(epoch, result, result_train)
        history.append(result_train) #result
    return history, train_recos
    
def testing(model, test_loader):
    results=[]
    reconstruction = []
    with torch.no_grad():
        for [batch] in test_loader: 
            batch=to_device(batch,device)
            h = model.encoder(batch) #, z_hat, mu, logvar
            h = h.squeeze()
            mu = model.mean(h)
            #logvar = model.log_var(h)
            #z_hat = model.reparametrize(mu, logvar)
            w = model.decoder(mu, (h, h))
            results.append(torch.mean((batch-w)**2,axis=1))
            reconstruction.append(w)
    return results, reconstruction
