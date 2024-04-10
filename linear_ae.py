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
    
class UsadModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder = Decoder(z_size, w_size)
  
  def training_step(self, batch, n):
    z = self.encoder(batch)
    w = self.decoder(z)
    loss = torch.mean((batch-w)**2)
    return loss

  def validation_step(self, batch, n):
    with torch.no_grad():
        z = self.encoder(batch)
        w = self.decoder(z)
        loss = torch.mean((batch-w)**2)
    return loss, w
        
  """def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    return {'val_loss1': epoch_loss1.item()}"""
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result))


def evaluate(model, val_loader, n):
    batch_loss = []
    outputs = []
    for [batch] in val_loader:
       batch = to_device(batch, device)
       loss, w = model.validation_step(batch, n)
       batch_loss.append(loss)
       outputs.append(w) 

    epoch_loss = torch.stack(batch_loss).mean()
    w_s = torch.stack(outputs)
    return epoch_loss, w_s
    

def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam): 
    history = []
    reconstructions = []
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()))
    
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=to_device(batch,device)
            
            #Train AE
            loss = model.training_step(batch,epoch+1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()            
            
        result, reconstruction = evaluate(model, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
        reconstructions.append(reconstruction)
    return history, reconstructions
    
def testing(model, test_loader, alpha=.5, beta=.5):
    results=[]
    with torch.no_grad():
        for [batch] in test_loader: 
            batch=to_device(batch,device)
            w = model.decoder(model.encoder(batch))
            results.append(torch.mean((batch-w)**2,axis=1))
    return results

def testing_prova(model, test_loader, alpha=.5, beta=.5):
    # QUI: farsi restituire anche w1 e w2 per fare il confronto con i valori originali
    # Attenzione: w1 e w2 sono calcolati per batch, quindi bisogna poi metterli tutti insieme
    # Problema: io sto passando il test_loader come sliding windows sovrapposte ---> perchè così funziona usad
    # Se però voglio visualizzare la ricostruzione, così non funziona più
    results=[]
    tensors_w = []
    with torch.no_grad():
        for [batch] in test_loader:
            batch=to_device(batch,device)
            w=model.decoder(model.encoder(batch))
            tensors_w.append(w)
            results.append(torch.mean((batch-w)**2,axis=1))
    return results, tensors_w

def reconstruction(model, test_loader):
  # QUI: il test loader che viene passato è ottenuto con non-overlapping sliding window
  tensors_w = []
  with torch.no_grad():
      for [batch] in test_loader: #N.B.: batch, w1, w2 sono tensori torch.tensor
          batch=to_device(batch,device)
          w=model.decoder(model.encoder(batch))
          tensors_w.append(w)
  # Restituisci solo le ricostruzioni da parte dei due autoencoder
  # Per determinare le anomalie: come facevamo con le baseline, da capire solo come mettere insieme i risultati del primo e del secondo decoder
  # Forse anche qui possiamo calcolare le loss, e almeno per il momento farne una media pesata... no?
  return tensors_w
