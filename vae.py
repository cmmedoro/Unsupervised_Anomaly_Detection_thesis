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
    #self.mean = nn.Linear(latent_size, latent_dim)
    #self.log_var = nn.Linear(latent_size, latent_dim)

  #def reparametrize(self, mu, logvar):
   #     std = torch.exp(0.5 * logvar)
    #    noise = torch.randn_like(std)
     #   noise = to_device(noise, device)

      #  z = mu + noise * std
       # return z

  def forward(self, w):
    #print("Input E: ", w.size())
    z, (h_n, c_n) = self.lstm(w)
    #print("Z: ", z.size())
    #print("H: ", h_n.size())
    # Prova con h_n al posto di z 
    #mu = self.mean(h_n)
    #print("Mu: ", mu.size())
    #logvar = self.log_var(h_n)
    #print("Var: ", logvar.size())
    #z_reparam = self.reparametrize(mu, logvar)
    #print("Z_reparametrized: ", z_reparam.size())
    return h_n#, z_reparam, mu, logvar
    
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
    self.output_layer = nn.Linear(latent_size, out_size)
        
  def forward(self, z, h):
    #print("Hidden state: ", h.size())
    #print("Z_reparametrized: ", z.size())
    batch = z.size()[1]
    n_feats = z.size()[2]
    #print("Input D: ", z.size())
    z = z.reshape((batch, n_feats))
    #h = h.reshape((batch, n_feats))
    #print("Reshaped input: ", z.size())
    #input = z.reshape((batch, self.latent_size))
    input = z.repeat(1, self.window)
    #print(input.size())
    input = input.reshape((batch, self.window, self.latent_size))
    #print(input.size(), h.size())
    #print(h.dim())
    w, (h_n, c_n) = self.lstm(input, h)
    #print("Out D: ", w.size())
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

        #kld_loss = torch.mean(
        #    -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        #)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return kld_loss
  
  def training_step(self, batch, criterion, n):
    h = self.encoder(batch) #, z_hat, mu, logvar
    mu = self.mean(h)
    logvar = self.log_var(h)
    z_hat = self.reparametrize(mu, logvar)
    w = self.decoder(z_hat, (h, h))
    loss_1 = criterion(w, batch)
    loss_2 = self.regularization_loss(mu, logvar)
    #print("Reconstruction loss: ",loss_1.size())
    #print("Regularization loss: ", loss_2.size())
    #loss = criterion(w, batch) + self.regularization_loss(mu, logvar)#torch.mean((batch-w)**2) #loss = mse
    loss = loss_1 + loss_2
    return loss, w

  def validation_step(self, batch, criterion, n):
    with torch.no_grad():
        h = self.encoder(batch) #, z_hat, mu, logvar
        mu = self.mean(h)
        logvar = self.log_var(h)
        z_hat = self.reparametrize(mu, logvar)
        w = self.decoder(z_hat, (h, h))
        loss_1 = criterion(w, batch)
        loss_2 = self.regularization_loss(mu, logvar)
        #print("Reconstruction loss: ",loss_1.size())
        #print("Regularization loss: ", loss_2.size())
        loss = loss_1 + loss_2#torch.mean((batch-w)**2) #loss = mse
    return loss
    
  def epoch_end(self, epoch, result, result_train):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, result_train, result))
    
def evaluate(model, val_loader, criterion, n):
    batch_loss = []
    for [batch] in val_loader:
       batch = to_device(batch, device)
       loss = model.validation_step(batch, criterion, n) 
       batch_loss.append(loss)

    epoch_loss = torch.stack(batch_loss).mean()
    return epoch_loss


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam): 
    history = []
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()))
    # Setup loss function
    criterion = nn.MSELoss().to(device)
    train_recos = []
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=to_device(batch,device)

            loss, train_reco = model.training_step(batch, criterion, epoch+1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_recos.append(train_reco)
            
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
            mu = model.mean(h)
            logvar = model.log_var(h)
            z_hat = model.reparametrize(mu, logvar)
            w = model.decoder(z_hat, (h, h))
            results.append(torch.mean((batch-w)**2,axis=1))
            reconstruction.append(w)
    return results, reconstruction, mu, logvar
