import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

from utils_ae import *
device = get_default_device()


class LstmModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
    super(LstmModel, self).__init__()
    """
    input_dim: number of features ---> 1
    hidden_dim: hidden_size of the LSTM layer ---> 50
    layer_dim: number of layers ---> 1
    output_dim: how many timestamps we want the model to predict ---> 1: the following one in the sequence
    """
    self.M = hidden_dim
    self.L = layer_dim

    self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True, dropout = 0.2)
    #batch_first to have (batch_dim, seq_dim, feature_dim)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, X):
    # initial hidden state and cell state
    h0 = torch.zeros(self.L, X.size(0), self.M).to(device)
    c0 = torch.zeros(self.L, X.size(0), self.M).to(device)

    out, (hn, cn) = self.rnn(X, (h0.detach(), c0.detach()))

    # h(T) at the final time step
    out = self.fc(self.relu(out[:, -1, :]))
    return out

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               device: torch.device
              ) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
  """
  model.train()
  train_loss, train_acc = 0, 0
  learning_rate = 0.001
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)

      y_pred = model(X)
      loss = loss_fn(y_pred, y)
      train_loss += loss.item()

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              device: torch.device) -> Tuple[float, float]:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
  """
  model.eval()

  test_loss, test_acc = 0, 0
  loss_fn = nn.MSELoss()

  with torch.inference_mode():
      for batch, (X, y) in enumerate(dataloader):
          X, y = X.to(device), y.to(device)

          test_pred_logits = model(X)

          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
    For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
  """
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          device=device)

      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          device=device)

      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  return results

"""class LstmModel(nn.Module):
  def __init__(self, in_size, latent_size): 
    super().__init__()
    
    in_size: number of features in input
    latent_size: size of the latent space of the lstm
    Ex. in_size = 5, latent_size = 50
    
    self.lstm = nn.LSTM(input_size=in_size, hidden_size=latent_size, num_layers=1, batch_first=True, dropout = 0.2
            # input and output tensors are provided as (batch, seq_len, feature(size))
        )
    self.relu = nn.ReLU()
    self.fc = nn.Linear(latent_size, 1)
  def forward(self, w):
    #print("Input: ", w.size())
    z, (h_n, c_n) = self.lstm(w)
    #print(z[:,-1, :].size())
    forecast = z[:, -1, :]
    forecast = self.relu(forecast)
    output = self.fc(forecast)
    #print("Output 3: ", output.size())
    return output
  
  def training_step(self, batch, criterion, n):
    z = self(batch)
    print("Z: ", z)
    print("Batch: ", batch)
    loss = criterion(z, batch)#torch.mean((batch-w)**2) #loss = mse
    return loss

  def validation_step(self, batch, y, criterion, n):
    with torch.no_grad():
        z = self(batch)
        loss = criterion(z, y)#torch.mean((batch-w)**2) #loss = mse
    return loss
    
  def epoch_end(self, epoch, result, result_train):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, result_train, result))
    
def evaluate(model, val_loader, criterion, n):
    batch_loss = []
    for X_batch, y_batch in val_loader:
       X_batch = to_device(X_batch, device)
       y_batch = to_device(y_batch, device)
       loss = model.validation_step(X_batch, y_batch, criterion, n) 
       batch_loss.append(loss)

    epoch_loss = torch.stack(batch_loss).mean()
    return epoch_loss


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam): 
    history = []
    optimizer = opt_func(model.parameters())
    criterion = nn.MSELoss().to(device) #nn.KLDivLoss(reduction="batchmean").to(device) #nn.MSELoss().to(device)
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = to_device(X_batch,device)
            y_batch = to_device(y_batch, device)
            optimizer.zero_grad()
            z = model(X_batch)
            #print("Z: ", z)
            #print("X_batch: ", X_batch)
            loss = criterion(z, y_batch)
            loss.backward()
            optimizer.step()
            
            
        result = evaluate(model, val_loader, criterion, epoch+1)
        result_train = evaluate(model, train_loader, criterion, epoch+1)
        model.epoch_end(epoch, result, result_train)
        #model.epoch_end(epoch, result)
        history.append(result)
    return history 
    
def testing(model, test_loader):
    results=[]
    forecast = []
    criterion = nn.MSELoss().to(device) #nn.KLDivLoss(reduction="batchmean").to(device)
    with torch.no_grad():
        for X_batch, y_batch in test_loader: 
            X_batch=to_device(X_batch,device)
            y_batch = to_device(y_batch, device)
            w=model(X_batch)
            results.append(criterion(w, y_batch))
            #results.append(torch.mean((batch-w)**2,axis=1))
            forecast.append(w)
    return results, forecast"""
