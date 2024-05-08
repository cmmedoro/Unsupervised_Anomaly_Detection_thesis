import torch
import torch.nn as nn

from utils_ae import *
device = get_default_device()

class Encoder(nn.Module):
  def __init__(self, in_size, latent_size): 
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True)

  def forward(self, w):
    out = self.linear1(w) #w
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    z = self.relu(out)
    return z
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.linear1(z)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    w = self.sigmoid(out)
    return w
    
class LinearVAE(nn.Module):
  def __init__(self, input_dim, latent_size, train_window): 
    super().__init__()
    self.encoder = Encoder(input_dim, latent_size)
    self.decoder = Decoder(latent_size, input_dim)
    self.mean = nn.Linear(latent_size, latent_size)
    self.log_var = nn.Linear(latent_size, latent_size)

  def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        noise = to_device(noise, device)

        z = mu + noise * std
        return z

  def regularization_loss(self, mu, logvar):
        kld_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return kld_loss
  
  def training_step(self, batch, criterion, n):
    z = self.encoder(batch)
    mu = self.mean(z)
    logvar = self.log_var(z)
    z_hat = self.reparametrize(mu, logvar)
    w = self.decoder(z_hat)
    loss_1 = criterion(w, batch)
    loss_2 = self.regularization_loss(mu, logvar)
    kld_weight = 0.00025 
    #print("Reconstruction loss: ",loss_1.size())
    #print("Regularization loss: ", loss_2.size())
    #loss = criterion(w, batch) + self.regularization_loss(mu, logvar)#torch.mean((batch-w)**2) #loss = mse
    loss = loss_1 + loss_2 * kld_weight
    return loss, w

  def validation_step(self, batch, criterion, n):
    with torch.no_grad():
        z = self.encoder(batch)
        mu = self.mean(z)
        logvar = self.log_var(z)
        z_hat = self.reparametrize(mu, logvar)
        w = self.decoder(z_hat)
        loss_1 = criterion(w, batch)
        loss_2 = self.regularization_loss(mu, logvar)
        kld_weight = 0.00025 
        loss = loss_1 + loss_2 * kld_weight#torch.mean((batch-w)**2) #loss = mse
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
        print("Epoch: ", epoch)
        for [batch] in train_loader:
            batch=to_device(batch,device)
            optimizer.zero_grad()

            loss, train_reco = model.training_step(batch, criterion, epoch+1)
            loss.backward()
            optimizer.step()
            
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
            z = model.encoder(batch)
            mu = model.mean(z)
            logvar = model.log_var(z)
            z_hat = model.reparametrize(mu, logvar)
            w = model.decoder(z_hat)
            results.append(torch.mean((batch-w)**2,axis=1))
            reconstruction.append(w)
    return results, reconstruction, mu, logvar
