import torch
import torch.nn as nn

from utils_ae import *
device = get_default_device()

class Encoder(nn.Module):
  def __init__(self, in_size, latent_size): 
    super().__init__()
    self.lstm = nn.LSTM(input_size=in_size, hidden_size=latent_size, num_layers=1, batch_first=True, dropout = 0.2
            # input and output tensors are provided as (batch, seq_len, feature(size))
        )
  def forward(self, w):
    #print("Input E: ", w.size())
    z, (h_n, c_n) = self.lstm(w)
    #print("Output E: ", h_n.size())
    # Here: do you return the output z or the last hidden state? Maybe the hidden state
    return h_n
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size): 
    super().__init__()
    self.latent_size = latent_size
    self.lstm = nn.LSTM(input_size=latent_size, hidden_size=latent_size, num_layers=1, batch_first=True, dropout = 0.2
            # input and output tensors are provided as (batch, seq_len, feature(size))
        )
    self.output_layer = nn.Linear(latent_size, out_size)
        
  def forward(self, z):
    batch = z.size()[1]
    n_feats = z.size()[2]
    #print("Input D: ", z.size())
    z = z.reshape((batch, n_feats))
    #print("Reshaped input: ", z.size())
    #input = z.reshape((batch, self.latent_size))
    input = z.repeat(1, 72)
    #print(input.size())
    input = input.reshape((batch, 72, self.latent_size))
    #print(input.size())
    w, (h_n, c_n) = self.lstm(input)
    #print("Out D: ", w.size())
    out = self.output_layer(w)
    #print("Output D: ", out.size())
    return out
    
class LstmAE(nn.Module):
  def __init__(self, input_dim, latent_size): 
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
    #w_s = outputs
    return epoch_loss#, w_s


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam): 
    history = []
    #eval_output = []
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()))
    # Setup loss function
    criterion = nn.MSELoss().to(device)
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=to_device(batch,device)

            loss = model.training_step(batch, criterion, epoch+1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
        result= evaluate(model, val_loader, criterion, epoch+1)
        model.epoch_end(epoch, result)
       # eval_output.append(w)
        history.append(result)
    return history  #, eval_output
    
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
