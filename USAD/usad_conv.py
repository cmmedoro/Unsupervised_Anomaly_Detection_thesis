import torch
import torch.nn as nn

from USAD.utils import *
device = get_default_device()

class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    # CONVOLUTIONAL ENCODER
    #in_channels = n_features
    self.conv1 = nn.Conv1d(in_channels=1, out_channels= 32, kernel_size=7, padding=3, stride=2)
    self.conv2 = nn.Conv1d(in_channels=32, out_channels= 16, kernel_size=7, padding=3, stride=2)
    self.conv3 = nn.Conv1d(in_channels=16, out_channels= 8, kernel_size=7, padding=3, stride=2)
    """
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    
    """
    self.relu = nn.ReLU(True)
    self.dropout = nn.Dropout(p=0.2)
  def forward(self, w):
    #print("Input encoder", w.size())
    #w = w.reshape((w.size()[0], w.size()[1], 1))
    out = self.conv1(w.permute(0, 2, 1)) #w #x.permute(0, 2, 1) ---> needed because conv1d wants input in form (batch, n_features, window_size)
    out = self.relu(out)
    out = self.dropout(out)
    #print("Conv1 encoder", out.size())
    out = self.conv2(out)
    out = self.relu(out)
    #print("Conv2 encoder", out.size())
    out = self.conv3(out)
    z = self.relu(out)
    #print("output encoder: ", z.size())
    return z
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.conv1 = nn.ConvTranspose1d(8, 16, 7, 2, 3 , 1)
    #self.conv2 = nn.ConvTranspose1d(16, 16, 7, 2, 3)
    self.conv3 = nn.ConvTranspose1d(16, 32, 7, 2, 3, 1)
    self.conv4 = nn.ConvTranspose1d(32, 1, 7, 2, 3, 1)
    """
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    """
    self.relu = nn.ReLU(True)
    self.dropout = nn.Dropout(p=0.2)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    #print("Input decoder: ", z.size())
    out = self.conv1(z)
    out = self.relu(out)
    #print("Conv1 decoder", out.size())
    #out = self.conv2(out)
    #out = self.relu(out)
    out = self.dropout(out)
    #print("Conv1 decoder", out.size())
    out = self.conv3(out)
    out = self.relu(out)
    #print("Conv3 decoder", out.size())
    out = self.conv4(out) 
    w = self.sigmoid(out)
    #print("Output decoder (before permutation): ", w.size())
    return w.permute(0, 2, 1)
    
class UsadModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
  
  def training_step(self, batch, n):
    #print("Batch: ", batch.size())
    z = self.encoder(batch)
    #print("Encoder output z: ", z.size())
    w1 = self.decoder1(z)
    #print("Decoder1 output w1: ", w1.size())
    w2 = self.decoder2(z)
    #print("Decoder output w2: ", w2.size())
    w3 = self.decoder2(self.encoder(w1))
    #print("Decoder output w3: ", w3.size())
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
  
  def validation_step_v2(self, batch, n):
    with torch.no_grad():
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
        loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return {'val_loss1': loss1, 'val_loss2': loss2}, w1, w3
        
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))
    
def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)

def evaluate_version2(model, val_loader, n):
    resulting_w1 = []
    resulting_w3 = []
    for [batch] in val_loader:
       outputs, w1, w3 = model.validation_step_v2(to_device(batch, device))
       resulting_w1.append(w1)
       resulting_w3.append(w3)
    return model.validation_epoch_end(outputs), resulting_w1, resulting_w3

def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam): #opt1 = None, opt2 = None, resume_training = False
    history = []
    # Changing code to allow resume training from checkpoint
    #if resume_training == False:
      # The following two lines were the original ones in the code
    optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    #else:
     #  optimizer1 = opt1
      # optimizer2 = opt2
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=to_device(batch,device)
            
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
            
            
        result = evaluate(model, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history#, optimizer1, optimizer2

def training_v2(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam): 
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch=to_device(batch,device)
            
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
            
            
        result, w1, w3 = evaluate_version2(model, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history, w1, w3
    
def testing(model, test_loader, alpha=.5, beta=.5):
    # QUI: farsi restituire anche w1 e w2 per fare il confronto con i valori originali
    # Attenzione: w1 e w2 sono calcolati per batch, quindi bisogna poi metterli tutti insieme
    results=[]
    with torch.no_grad():
        for [batch] in test_loader: #N.B.: batch, w1, w2 sono tensori torch.tensor
            batch=to_device(batch,device)
            w1=model.decoder1(model.encoder(batch))
            w2=model.decoder2(model.encoder(w1))
            results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
    return results

def testing_prova(model, test_loader, alpha=.5, beta=.5):
    # QUI: farsi restituire anche w1 e w2 per fare il confronto con i valori originali
    # Attenzione: w1 e w2 sono calcolati per batch, quindi bisogna poi metterli tutti insieme
    # Problema: io sto passando il test_loader come sliding windows sovrapposte ---> perchè così funziona usad
    # Se però voglio visualizzare la ricostruzione, così non funziona più
    results=[]
    tensors_w1 = []
    tensors_w2 = []
    with torch.no_grad():
        for [batch] in test_loader:
            batch=to_device(batch,device)
            w1=model.decoder1(model.encoder(batch))
            w2=model.decoder2(model.encoder(w1))
            tensors_w1.append(w1)
            tensors_w2.append(w2)
            results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
    return results, tensors_w1, tensors_w2

def reconstruction(model, test_loader):
  # QUI: il test loader che viene passato è ottenuto con non-overlapping sliding window
  with torch.no_grad():
      for [batch] in test_loader: #N.B.: batch, w1, w2 sono tensori torch.tensor
          batch=to_device(batch,device)
          w1=model.decoder1(model.encoder(batch))
          w2=model.decoder2(model.encoder(w1))
  # Restituisci solo le ricostruzioni da parte dei due autoencoder
  # Per determinare le anomalie: come facevamo con le baseline, da capire solo come mettere insieme i risultati del primo e del secondo decoder
  # Forse anche qui possiamo calcolare le loss, e almeno per il momento farne una media pesata... no?
  return w1, w2
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