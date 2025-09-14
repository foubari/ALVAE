import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT
from .base_vae import BaseVAE


class StudentTVAE(BaseVAE):
    """Student-t VAE for robust density estimation"""
    
    def __init__(self, encoder, decoder, latent_dim, beta=1.0):
        super(StudentTVAE, self).__init__(encoder, decoder, latent_dim)
        self.beta = beta
        
        # Modify encoder to output df parameter
        if hasattr(self.encoder, 'add_df_output'):
            self.encoder.add_df_output()
            
    def encode(self, x):
        """Encode to Student-t parameters"""
        if hasattr(self.encoder, 'encode_with_extras'):
            mu, logvar, extras = self.encoder.encode_with_extras(x)
            df = extras.get('df', torch.ones_like(mu) * 5.0)  # Default df=5
        else:
            mu, logvar = self.encoder(x)
            df = torch.ones_like(mu) * 5.0
            
        return mu, logvar, df
    
    def decode(self, z):
        """Decode latent codes"""
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar, df):
        """Student-t reparameterization"""
        std = torch.exp(0.5 * logvar)
        
        # Sample from Student-t using Gaussian and Chi-squared
        eps = torch.randn_like(mu)
        chi2 = torch.distributions.Chi2(df).rsample()
        chi2 = chi2.unsqueeze(-1).expand_as(mu)
        
        # Student-t = Gaussian / sqrt(Chi2/df)
        z = mu + std * eps * torch.sqrt(df / chi2)
        
        return z
    
    def forward(self, x):
        mu, logvar, df = self.encode(x)
        z = self.reparameterize(mu, logvar, df)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, df, z
    
    def loss_function(self, x, x_recon, mu, logvar, df, **kwargs):
        """
        Compute ELBO with Student-t prior and posterior
        """
        batch_size = x.size(0)
        
        # Reconstruction loss with Student-t likelihood
        precision = torch.exp(-logvar)
        diff = x - x_recon
        
        # Student-t log likelihood
        recon_loss = -torch.sum(
            torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2) 
            - 0.5 * torch.log(np.pi * df) - 0.5 * logvar
            - (df + 1) / 2 * torch.log(1 + precision * diff**2 / df)
        ) / batch_size
        
        # KL divergence between two Student-t distributions
        # Using approximation for tractability
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'df': df.mean()
        }
    
    def sample_prior(self, num_samples, device):
        """Sample from Student-t prior"""
        df = 5.0  # Default degrees of freedom
        dist = StudentT(df, torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        return dist.sample((num_samples,)).to(device)