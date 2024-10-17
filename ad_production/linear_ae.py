import torch
import torch.nn as nn

from utils_ae import get_default_device, to_device
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
    
class LinearAE(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.z_size = 9
    self.encoder = Encoder(w_size, self.z_size)
    self.decoder = Decoder(self.z_size, 72) #w_size
  
  def training_step(self, batch, y, criterion, n):
    z = self.encoder(batch)
    w = self.decoder(z)
    #w_n = w.reshape(w.size()[0], w.size()[1], 1)
    #batch_n = batch.reshape(batch.size()[0], 72, int(batch.size()[1]/72))
    #batch_nn = batch_n[:, :, 0].unsqueeze(-1)
    #batch_n = y.reshape
    loss = criterion(w, y)
    #loss = criterion(w, batch) #torch.mean((batch-w)**2)#
    return loss

  def validation_step(self, batch, y, criterion, n):
    with torch.no_grad():
        z = self.encoder(batch)
        w = self.decoder(z)
        #w_n = w.reshape(w.size()[0], w.size()[1], 1)
        #batch_n = batch.reshape(batch.size()[0], 72, int(batch.size()[1]/72))
        #batch_nn = batch_n[:, :, 0].unsqueeze(-1)
        loss = criterion(w, y)#criterion(w_n, batch_nn)
        #loss = criterion(w, batch) #torch.mean((batch-w)**2) 
    return loss
        
  """def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    return {'val_loss1': epoch_loss1.item()}"""
    
  def epoch_end(self, epoch, result, result_train):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, result_train, result))
    #print("Epoch [{}], val_loss1: {:.4f}".format(epoch, result))


def evaluate(model, val_loader, criterion, device, n):
    batch_loss = []
    #for [batch] in val_loader:
    for X_batch, y_batch in val_loader:
       #batch = batch.to(device) #to_device(batch, device)
       X_batch = X_batch.to(device) #to_device(X_batch,device)
       y_batch = y_batch.to(device) #to_device(y_batch, device)

       #loss = model.validation_step(batch, criterion, n) 
       loss = model.validation_step(X_batch, y_batch, criterion, n)
       batch_loss.append(loss)

    epoch_loss = torch.stack(batch_loss).mean()
    return epoch_loss.item()
    

def training(epochs, model, train_loader, val_loader, device, opt_func=torch.optim.Adam): 
    history = []
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()))
    criterion = nn.MSELoss().to(device)#nn.KLDivLoss(reduction="batchmean").to(device) #nn.MSELoss().to(device)
    #criterion = to_device(criterion, device)
    for epoch in range(epochs):
        train_loss = []
        #for [batch] in train_loader:
        for X_batch, y_batch in train_loader:
            #batch = batch.to(device)#to_device(batch,device)
            X_batch = X_batch.to(device) #to_device(X_batch,device)
            y_batch = y_batch.to(device) #to_device(y_batch, device)

            optimizer.zero_grad() 

            #Train AE
            #loss = model.training_step(batch, criterion, epoch+1)
            loss = model.training_step(X_batch, y_batch, criterion, epoch+1)
            train_loss.append(loss)
            loss.backward()
            optimizer.step()
            #optimizer.zero_grad()            
        result_train = torch.stack(train_loss).mean()     
        result = evaluate(model, val_loader, criterion, device, epoch+1) 
        model.epoch_end(epoch, result, result_train)
        res = result_train.item()
        history.append((res, result))
    return history
    
def testing(model, test_loader, device):
    results=[]
    reconstruction = []
    with torch.no_grad():
        for [batch] in test_loader:
        #for X_batch, y_batch in test_loader: 
            batch = batch.to(device) #to_device(batch,device)
            #X_batch = X_batch.to(device) #to_device(X_batch,device)
            #y_batch = y_batch.to(device) #to_device(y_batch, device)

            w=model.decoder(model.encoder(batch))
            #batch_s = batch[:, :, 0]
            #w_s = w.reshape(w.size()[0], w.size()[1], 1)
            #batch_n = batch.reshape(batch.size()[0], 72, int(batch.size()[1]/72))
            #batch_s = batch_n[:, :, 0].unsqueeze(-1)
            #batch_s = batch_s.reshape(batch.size()[0], batch.size()[1], 1)
            #w_s = w
            #results.append(criterion(w, batch))
            results.append(torch.mean((batch-w)**2,axis=1))

            #results.append(torch.mean((batch-w)**2,axis=1))
            reconstruction.append(w)
    return results, reconstruction

