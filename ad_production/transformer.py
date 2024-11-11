import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class Transformer(nn.Module):
    def __init__(self, input_features = 1, d_model = 32, dim_feedforward = 256, num_layers = 1, window_length = 48, n_head = 1, positional_encoding = None):
        super(Transformer, self).__init__()
        self.seq_len = window_length # definisci la lunghezza della window
        self.d_model = d_model # dimensione del modello
        self.dim_feedforward = dim_feedforward # dimensione dei layer ff
        self.num_layers = num_layers #quantità di layer di encoder del transformer
        self.nhead = n_head # numero di teste del transformer --> multi-head attention
        self.positional_encoding = positional_encoding # se aggiungiamo un positional encoding o meno
        # Il modello si costituisce di:
        # - Linear layer iniziale che serve come encoding della sequenza in input
        # - positional encoding da aggiungere
        # - encoder costituito da encoder layer
        # - linear layer finale caratterizzato da una sigmoide per dare gli output fra 0 e 1 (l'output è la predizione per lo step successivo alla finestra)
        self.embedding = nn.Linear(input_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                                   dim_feedforward=self.dim_feedforward, dropout=0.1,
                                                   activation='relu', layer_norm_eps=1e-05,
                                                   batch_first=True, norm_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.fc = nn.Linear(self.d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # Pass the input through the embedding layers
      # I = (batch_size, window_lenght, n_features)
      #print("Input: ", x.size())
      emd_x = self.embedding(x)
      #print("Embedding: ", emd_x.size())
      # Pass it through the encoder layer
      enc_x = self.encoder(emd_x)
      #print("Encoder: ", enc_x.size())
      # Pass it through a linear layer to output a single value, for the next step in the sequence
      last_output = enc_x[:, -1, :]
      #print("Last output: ", last_output.size())
      out = self.fc(last_output) #self.fc(enc_x)
      out = self.sigmoid(out)
      #print("Output: ", out.size())
      return out, enc_x
    

def training(num_epochs, model, train_loader, eval_loader, device, optimizer = torch.optim.Adam):
    criterion = nn.MSELoss().to(device)
    loss_history = []
    optimizer = optimizer(model.parameters())
    for epoch in range(num_epochs):
        ### TRAINING ###
        model.train()
        total_loss = 0
        for inputs, target in train_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            #inputs = inputs.unsqueeze(2)  # Add feature dimension (batch_size, seq_len, input_size)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.squeeze(), target) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        ### VALIDATING ###
        model.eval()
        val_loss = 0
        for X, y in eval_loader:
            X = X.to(device)
            y = y.to(device)
            with torch.no_grad():
              outputs, _ = model(X)
              loss = criterion(outputs.squeeze(), y)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(eval_loader)

        loss_history.append((avg_loss, avg_val_loss))
        print(f'Epoch {epoch+1}/{num_epochs}, Train loss: {avg_loss}, Validation loss: {avg_val_loss}')


def testing(model, test_loader, device):
    results=[]
    forecast = []
    criterion = nn.MSELoss().to(device)
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w, _ = model(X_batch)
            results.append(criterion(w.squeeze(), y_batch))
            forecast.append(w)
    return results, forecast