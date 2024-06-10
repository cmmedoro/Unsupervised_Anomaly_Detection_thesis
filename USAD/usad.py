import torch
import torch.nn as nn
from tqdm import tqdm

#from USAD.utils import *
#device = get_default_device()
#device = torch.device("cuda")

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
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
  
  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2

  def validation_step(self, batch, n):
    with torch.no_grad():
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
        loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return {'val_loss1': loss1, 'val_loss2': loss2}
        
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))
    
def evaluate(model, val_loader, device, n):
    #outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]
    outputs = [model.validation_step(batch.to(device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)

def training(epochs, model, train_loader, val_loader, device, opt_func=torch.optim.Adam): 
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch= batch.to(device)  #to_device(batch,device)
            
            #Train AE1
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            
            #Train AE2
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
            
        result = evaluate(model, val_loader, device, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
    
def testing(model, test_loader, device, alpha=.5, beta=.5):
    # QUI: farsi restituire anche w1 e w2 per fare il confronto con i valori originali
    # Attenzione: w1 e w2 sono calcolati per batch, quindi bisogna poi metterli tutti insieme
    results=[]
    with torch.no_grad():
        for [batch] in test_loader: #N.B.: batch, w1, w2 sono tensori torch.tensor
            batch= batch.to(device) #to_device(batch,device)
            w1=model.decoder1(model.encoder(batch))
            w2=model.decoder2(model.encoder(w1))
            results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
    return results

def testing_prova(model, test_loader, device, alpha=.5, beta=.5):
    # QUI: farsi restituire anche w1 e w2 per fare il confronto con i valori originali
    # Attenzione: w1 e w2 sono calcolati per batch, quindi bisogna poi metterli tutti insieme
    # Problema: io sto passando il test_loader come sliding windows sovrapposte ---> perchè così funziona usad
    # Se però voglio visualizzare la ricostruzione, così non funziona più
    results=[]
    tensors_w1 = []
    tensors_w2 = []
    with torch.no_grad():
        for [batch] in test_loader:
            batch=batch.to(device) #to_device(batch,device)
            w1=model.decoder1(model.encoder(batch))
            w2=model.decoder2(model.encoder(w1))
            tensors_w1.append(w1)
            tensors_w2.append(w2)
            results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
    return results, tensors_w1, tensors_w2

def reconstruction(model, test_loader, device):
  # QUI: il test loader che viene passato è ottenuto con non-overlapping sliding window
  tensors_w1 = []
  tensors_w2 = []
  with torch.no_grad():
      for [batch] in test_loader: #N.B.: batch, w1, w2 sono tensori torch.tensor
          batch= batch.to(device) #to_device(batch,device)
          w1=model.decoder1(model.encoder(batch))
          w2=model.decoder2(model.encoder(w1))
          tensors_w1.append(w1)
          tensors_w2.append(w2)
  # Restituisci solo le ricostruzioni da parte dei due autoencoder
  # Per determinare le anomalie: come facevamo con le baseline, da capire solo come mettere insieme i risultati del primo e del secondo decoder
  # Forse anche qui possiamo calcolare le loss, e almeno per il momento farne una media pesata... no?
  return tensors_w1, tensors_w2
"""
Codice per ricominciare il training dal checkpoint
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

Load the model like this:
  model = MyModel(*args, **kwargs)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  ckp_path = "path/to/checkpoint/checkpoint.pt"
  model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)

Then pass the model to the training loop, it should resume from where it ended



FROM PYTORCH documentation
Load several info
# Additional information
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4

torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

Load general checkpoint: firstly you need to instantiate model and optimizer, then load the state dictionary
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
#Resume training as follows (similarly if you want to resume evaluation):
model.train() #or model.eval()
"""