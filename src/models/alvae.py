import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_vae import BaseVAE


class AsymmetricLaplaceVAE(BaseVAE):
    """Asymmetric Laplace VAE for skewed and heavy-tailed data"""
    
    def __init__(self, encoder, decoder, latent_dim, beta=1.0, mmd_weight=1.0):
        super(AsymmetricLaplaceVAE, self).__init__(encoder, decoder, latent_dim)
        self.beta = beta
        self.mmd_weight = mmd_weight
        
        # Modify encoder to output skew parameter
        if hasattr(self.encoder, 'add_skew_output'):
            self.encoder.add_skew_output()
            
    def encode(self, x):
        """Encode to Asymmetric Laplace parameters"""
        if hasattr(self.encoder, 'encode_with_extras'):
            mu, logvar, extras = self.encoder.encode_with_extras(x)
            skew = extras.get('skew', torch.zeros_like(mu))
        else:
            mu, logvar = self.encoder(x)
            skew = torch.zeros_like(mu)
            
        # Convert to AL parameters
        scale = torch.exp(0.5 * logvar)
        
        return mu, scale, skew
    
    def decode(self, z):
        """Decode latent codes"""
        return self.decoder(z)
    
    def sample_asymmetric_laplace(self, mu, scale, skew):
        """
        Sample from Asymmetric Laplace distribution
        AL(μ, σ, κ) where κ is the skewness parameter
        """
        batch_size = mu.size(0)
        device = mu.device
        
        # Sample from uniform distribution
        u = torch.rand_like(mu)
        
        # Convert skew to asymmetry parameters
        p = torch.sigmoid(skew)  # Probability of positive side
        
        # Scale parameters for each side
        scale_left = scale / (2 * (1 - p + 1e-8))
        scale_right = scale / (2 * (p + 1e-8))
        
        # Sample from Asymmetric Laplace
        z = torch.where(
            u < (1 - p),
            mu + scale_left * torch.log(u / (1 - p + 1e-8)),
            mu - scale_right * torch.log((1 - u) / (p + 1e-8))
        )
        
        return z
    
    def forward(self, x):
        mu, scale, skew = self.encode(x)
        z = self.sample_asymmetric_laplace(mu, scale, skew)
        x_recon = self.decode(z)
        return x_recon, mu, scale, skew, z
    
    def compute_mmd(self, z, prior_samples):
        """
        Compute Maximum Mean Discrepancy between posterior and prior
        Using RBF kernel
        """
        def rbf_kernel(x, y, bandwidth=1.0):
            """RBF kernel computation"""
            xx = torch.mm(x, x.t())
            yy = torch.mm(y, y.t())
            xy = torch.mm(x, y.t())
            
            rx = torch.diag(xx).unsqueeze(0).expand_as(xx)
            ry = torch.diag(yy).unsqueeze(0).expand_as(yy)
            
            dxx = rx.t() + rx - 2 * xx
            dyy = ry.t() + ry - 2 * yy
            dxy = rx.t() + ry - 2 * xy
            
            kernel_val = torch.exp(-dxx / (2 * bandwidth**2))
            kernel_val += torch.exp(-dyy / (2 * bandwidth**2))
            kernel_val -= 2 * torch.exp(-dxy / (2 * bandwidth**2))
            
            return kernel_val.mean()
        
        # Use multiple bandwidths for better MMD estimation
        bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]
        mmd = 0.0
        
        for bw in bandwidths:
            mmd += rbf_kernel(z, prior_samples, bw)
            
        return mmd / len(bandwidths)
    
    def loss_function(self, x, x_recon, mu, scale, skew, z, **kwargs):
        """
        Compute ELBO with MMD divergence for Asymmetric Laplace
        """
        batch_size = x.size(0)
        device = x.device
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
        
        # Sample from prior for MMD
        prior_samples = self.sample_prior(batch_size, device)
        
        # MMD divergence
        mmd_loss = self.compute_mmd(z, prior_samples)
        
        # Total loss
        loss = recon_loss + self.mmd_weight * mmd_loss
        
        # Additional metrics
        skewness = skew.mean()
        kurtosis = ((z - z.mean()) ** 4).mean() / (z.var() ** 2)
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'mmd_loss': mmd_loss,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def sample_prior(self, num_samples, device):
        """Sample from Asymmetric Laplace prior"""
        mu = torch.zeros(num_samples, self.latent_dim).to(device)
        scale = torch.ones(num_samples, self.latent_dim).to(device)
        skew = torch.zeros(num_samples, self.latent_dim).to(device)
        
        return self.sample_asymmetric_laplace(mu, scale, skew)