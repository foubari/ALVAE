import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Poisson
from .base_vae import BaseVAE


class PoissonVAE(BaseVAE):
    """Poisson VAE with discrete spike-count latents"""
    
    def __init__(self, encoder, decoder, latent_dim, beta=1.0, temperature=1.0):
        super(PoissonVAE, self).__init__(encoder, decoder, latent_dim)
        self.beta = beta
        self.temperature = temperature
        
    def encode(self, x):
        """Encode to Poisson rate parameters"""
        mu, logvar = self.encoder(x)
        # Convert to positive rates
        rates = F.softplus(mu) + 1e-6
        return rates
    
    def decode(self, z):
        """Decode discrete latent codes"""
        return self.decoder(z)
    
    def poisson_reparameterization(self, rates):
        """
        Poisson reparameterization using continuous relaxation
        Based on the Gumbel-Softmax trick adapted for Poisson
        """
        # Sample from exponential distribution
        eps = torch.rand_like(rates) + 1e-8
        exp_samples = -torch.log(eps)
        
        # Cumulative sum for Poisson process
        cumsum = torch.cumsum(exp_samples / rates, dim=-1)
        
        # Apply sigmoid for continuous relaxation
        z = torch.sigmoid((1.0 - cumsum) / self.temperature)
        
        # During evaluation, discretize
        if not self.training:
            z = torch.floor(rates)
            
        return z
    
    def forward(self, x):
        rates = self.encode(x)
        z = self.poisson_reparameterization(rates)
        x_recon = self.decode(z)
        return x_recon, rates, z
    
    def loss_function(self, x, x_recon, rates, z, **kwargs):
        """
        Compute ELBO with Poisson prior
        KL(Poisson(λ) || Poisson(λ0)) ≈ λ log(λ/λ0) - (λ - λ0)
        """
        batch_size = x.size(0)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
        
        # KL divergence between Poisson distributions
        prior_rate = 1.0  # Prior rate parameter
        kl_loss = torch.sum(
            rates * torch.log(rates / prior_rate) - (rates - prior_rate)
        ) / batch_size
        
        # Sparsity penalty (metabolic cost)
        sparsity_loss = torch.mean(rates)
        
        loss = recon_loss + self.beta * kl_loss + 0.1 * sparsity_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'sparsity': sparsity_loss,
            'active_units': (z > 0.1).float().mean()
        }
    
    def sample_prior(self, num_samples, device):
        """Sample from Poisson prior"""
        prior_rate = 1.0
        dist = Poisson(torch.ones(num_samples, self.latent_dim) * prior_rate)
        return dist.sample().float().to(device)