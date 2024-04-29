import torch
import torch.nn as nn
from tqdm import tqdm

from utils_ae import *
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
    #print("Input E: ", w.size())
    out = self.conv1(w.permute(0, 2, 1)) #w #x.permute(0, 2, 1) ---> needed because conv1d wants input in form (batch, n_features, window_size)
    #print("Conv1 E: ", out.size())
    out = self.relu(out)
    out = self.dropout(out)
    out = self.conv2(out)
    #print("Conv2 E: ", out.size())
    out = self.relu(out)
    out = self.conv3(out)
    #print("Conv3 E: ", out.size())
    z = self.relu(out)
    #print("Output E: ", z.size())
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
    #print("Input D: ", z.size())
    out = self.conv1(z)
    #print("Conv1 D: ", out.size())
    out = self.relu(out)
    out = self.dropout(out)
    out = self.conv3(out)
    #print("Conv2 D: ", out.size())
    out = self.relu(out)
    out = self.conv4(out) 
    #print("Conv3 D: ", z.size())
    w = self.sigmoid(out)
    #print("Output D: ", w.size())
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
    return loss#, w
        
  """def validation_epoch_end(self, outputs):
    batch_losses = [x for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    return {'val_loss': epoch_loss.item()}"""
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss: {:.4f}".format(epoch, result))
    
def evaluate(model, val_loader, criterion, n):
    batch_loss = []
    #outputs = []
    for [batch] in val_loader:
       batch = to_device(batch, device)
       loss = model.validation_step(batch, criterion, n) #, w
       batch_loss.append(loss)
       #outputs.append(w) 

    epoch_loss = torch.stack(batch_loss).mean()
    #w_s = torch.stack(outputs)
   # w_s = outputs
    return epoch_loss#, w_s


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam): 
    history = []
    #eval_output = []
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()))
    # Setup loss function
    criterion = nn.KLDivLoss(reduction="batchmean").to(device) #nn.MSELoss().to(device)
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=to_device(batch,device)

            loss = model.training_step(batch, criterion, epoch+1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
        result = evaluate(model, val_loader, criterion, epoch+1) #
        model.epoch_end(epoch, result)
        #eval_output.append(w)
        history.append(result)
    return history#, eval_output
    
def testing(model, test_loader):
    results=[]
    reconstruction = []
    criterion = nn.KLDivLoss(reduction="batchmean").to(device) #nn.MSELoss().to(device)
    with torch.no_grad():
        for [batch] in test_loader: 
            batch=to_device(batch,device)
            w=model.decoder(model.encoder(batch))
            results.append(criterion(w, batch))
            #results.append(torch.mean((batch-w)**2,axis=1))
            reconstruction.append(w)
    return results, reconstruction
