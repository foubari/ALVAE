import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_vae import BaseVAE


class GaussianVAE(BaseVAE):
    """Standard Gaussian VAE implementation"""
    
    def __init__(self, encoder, decoder, latent_dim, beta=1.0):
        super(GaussianVAE, self).__init__(encoder, decoder, latent_dim)
        self.beta = beta  # KL weight for beta-VAE
        
    def encode(self, x):
        """Encode to Gaussian parameters"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent codes"""
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        """Gaussian reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
    
    def loss_function(self, x, x_recon, mu, logvar, **kwargs):
        """
        Compute ELBO loss
        KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        """
        batch_size = x.size(0)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Total loss
        loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample_prior(self, num_samples, device):
        """Sample from standard Gaussian prior"""
        return torch.randn(num_samples, self.latent_dim).to(device)