import torch
import torch.nn as nn
import math

from utils_ae import *

device = get_default_device()

### To finish ###
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, data_type="real"):
        super(VAE, self).__init__()
        # Encoder: layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        # Decoder: layers
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc41 = nn.Linear(hidden_size, input_size)
        self.fc42 = nn.Linear(hidden_size, input_size)
        # data_type: can be "binary" or "real"
        self.data_type = data_type

    def encode(self, x):
        h1 = torch.tanh(self.fc1(x))
        mean, log_var = self.fc21(h1), self.fc22(h1)
        return mean, log_var

    @staticmethod
    def reparameterize(mean, log_var):
        mu, sigma = mean, torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        return z

    def decode(self, z):
        h3 = torch.tanh(self.fc3(z))
        if self.data_type == "real":
            mean, log_var = torch.sigmoid(self.fc41(h3)), self.fc42(h3)
            return mean, log_var
        else:
            logits = self.fc41(h3)
            probs = torch.sigmoid(logits)
            return probs

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z_mean, z_logvar, self.decode(z)
    
"""
QUESTO VA BENE SE I DATI SONO DI TIPO BINARY
def compute_elbo(x, reconst_x, mean, log_var):
    # ELBO(Evidence Lower Bound) is the objective of VAE, we train the model just to maximize the ELBO.
    reconst_error = -torch.nn.functional.mse_loss(reconst_x, x, reduction='sum')
    # see Appendix B from VAE paper: "Kingma and Welling. Auto-Encoding Variational Bayes. ICLR-2014."
    # -KL[q(z|x)||p(z)] = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    elbo = (reconst_error - kl_divergence) / len(x)
    return elbo
"""
pi = torch.Tensor([math.pi])
def compute_elbo(x, mean, log_var, out_mean, out_logvar):
    # ELBO(Evidence Lower Bound) is the objective of VAE, we train the model just to maximize the ELBO.
    
    reconst_error = -0.5 * torch.sum(torch.log(2*pi) + out_logvar + (x - out_mean).pow(2) / out_logvar.exp())
    # see Appendix B from VAE paper: "Kingma and Welling. Auto-Encoding Variational Bayes. ICLR-2014."
    # -KL[q(z|x)||p(z)] = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    elbo = (reconst_error - kl_divergence) / len(x)
    return elbo

def training(model, epochs, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr = 1e-3) 
    for epoch in range(epochs):
        loss = 0
        for [batch_features] in train_loader:
            batch_features = batch_features.to(device)
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            # compute reconstructions
            mu, logvar, (out_mu, out_var) = model(batch_features)
            train_loss = -compute_elbo(batch_features, mu, logvar, out_mu, out_var)
            # compute accumulated gradients
            train_loss.backward()
            # perform parameter update based on current gradients
            optimizer.step()
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        # compute the epoch training loss
        loss = loss / len(train_loader)
        # display the epoch training loss
        #print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        ### EVALUATION ###
        eval_loss = 0
        for [batch] in val_loader:
            batch = batch.to(device)
            with torch.no_grad():
                mu_v, logvar_v, (out_mu_v, out_var_v) = model(batch)
                val_loss = -compute_elbo(batch, mu_v, logvar_v, out_mu_v, out_var_v)
                eval_loss += val_loss.item()
        eval_loss = eval_loss / len(val_loader)
        # display the epoch training and validation loss
        print("epoch : {}/{}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch + 1, epochs, loss, eval_loss))
        history.append({'train_loss': loss, 'eval_loss': eval_loss})
    return history

def testing(model, test_loader):
    results=[]
    reconstructions = []
    with torch.no_grad():
        for [batch] in test_loader: 
            batch=to_device(batch,device)
            mu, logvar, (out_mu, out_var) = model(batch)
            reconstruction = out_mu
            results.append(torch.mean((batch-reconstruction)**2,axis=1))
            reconstructions.append(reconstruction)
    return results, reconstructions, mu, logvar