import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseVAE(nn.Module, ABC):
    """Base class for all VAE models"""
    
    def __init__(self, encoder, decoder, latent_dim):
        super(BaseVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        
    @abstractmethod
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        pass
    
    @abstractmethod
    def decode(self, z):
        """Decode latent codes to reconstruction"""
        pass
    
    @abstractmethod
    def reparameterize(self, *params):
        """Reparameterization trick for sampling"""
        pass
    
    @abstractmethod
    def forward(self, x):
        """Forward pass through the model"""
        pass
    
    @abstractmethod
    def loss_function(self, x, x_recon, *params):
        """Compute VAE loss"""
        pass
    
    def sample(self, num_samples, device):
        """Sample from the prior and decode"""
        z = self.sample_prior(num_samples, device)
        return self.decode(z)
    
    @abstractmethod
    def sample_prior(self, num_samples, device):
        """Sample from the prior distribution"""
        pass