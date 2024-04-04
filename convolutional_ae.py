import torch
import torch.nn as nn

from utils import *
device = get_default_device()

class Encoder(nn.Module):
  def __init__(self, n_features, latent_size): #(1, 32)
    super().__init__()
    # CONVOLUTIONAL ENCODER
    #in_channels = n_features
    self.conv1 = nn.Conv1d(in_channels=n_features, out_channels= latent_size, kernel_size=7, padding=3, stride=2)
    self.conv2 = nn.Conv1d(in_channels=latent_size, out_channels= latent_size//2, kernel_size=7, padding=3, stride=2)
    self.conv3 = nn.Conv1d(in_channels=latent_size//2, out_channels= latent_size//4, kernel_size=7, padding=3, stride=2)
    self.relu = nn.ReLU(True)
    self.dropout = nn.Dropout(p=0.2)
  def forward(self, w):
    out = self.conv1(w.permute(0, 2, 1)) #w #x.permute(0, 2, 1) ---> needed because conv1d wants input in form (batch, n_features, window_size)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.conv2(out)
    out = self.relu(out)
    out = self.conv3(out)
    z = self.relu(out)
    return z
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size): #(32, 1)
    super().__init__()
    self.conv1 = nn.ConvTranspose1d(latent_size//4, latent_size//2, 7, 2, 3 , 1)
    self.conv3 = nn.ConvTranspose1d(latent_size//2, latent_size, 7, 2, 3, 1)
    self.conv4 = nn.ConvTranspose1d(latent_size, out_size, 7, 2, 3, 1)
    self.relu = nn.ReLU(True)
    self.dropout = nn.Dropout(p=0.2)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.conv1(z)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.conv3(out)
    out = self.relu(out)
    out = self.conv4(out) 
    w = self.sigmoid(out)
    return w.permute(0, 2, 1)
    
class ConvAE(nn.Module):
  def __init__(self, input_dim, latent_size): #(1, 32)
    super().__init__()
    self.encoder = Encoder(input_dim, latent_size)
    self.decoder = Decoder(latent_size, input_dim)
  
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
    return loss, w
        
  def validation_epoch_end(self, outputs):
    batch_losses = [x for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    return {'val_loss': epoch_loss.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))
    
def evaluate(model, val_loader, criterion, n):
    outputs = [model.validation_step(to_device(batch,device), criterion, n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam): 
    history = []
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    # Setup loss function
    criterion = nn.MSELoss().to(device)
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=to_device(batch,device)

            loss = model.training_step(batch, criterion, epoch+1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
        result, w = evaluate(model, val_loader, criterion, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history, w
    
def testing(model, test_loader):
    results=[]
    reconstruction = []
    with torch.no_grad():
        for [batch] in test_loader: 
            batch=to_device(batch,device)
            w=model.decoder(model.encoder(batch))
            results.append(torch.mean((batch-w)**2,axis=1))
            reconstruction.append(w)
    return results, reconstruction
