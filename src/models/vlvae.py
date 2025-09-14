import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_vae import BaseVAE


class VLVAE(BaseVAE):
    """Variational Laplace VAE with full covariance"""
    
    def __init__(self, encoder, decoder, latent_dim, num_iterations=5, beta=1.0):
        super(VLVAE, self).__init__(encoder, decoder, latent_dim)
        self.beta = beta
        self.num_iterations = num_iterations
        
    def encode(self, x):
        """Initial encoding to get starting point"""
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def decode(self, z):
        """Decode latent codes"""
        return self.decoder(z)
    
    def compute_jacobian(self, x, z):
        """Compute Jacobian of decoder w.r.t. z"""
        z = z.requires_grad_(True)
        x_recon = self.decode(z)
        
        batch_size = z.size(0)
        latent_dim = z.size(1)
        output_dim = x_recon.size(1)
        
        jacobian = torch.zeros(batch_size, output_dim, latent_dim).to(z.device)
        
        for i in range(output_dim):
            grad_outputs = torch.zeros_like(x_recon)
            grad_outputs[:, i] = 1.0
            
            grads = torch.autograd.grad(
                outputs=x_recon,
                inputs=z,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=True
            )[0]
            
            jacobian[:, i, :] = grads
            
        return jacobian
    
    def iterative_mode_update(self, x, mu_init, num_iter=None):
        """Iteratively update the mode of posterior"""
        if num_iter is None:
            num_iter = self.num_iterations
            
        mu = mu_init
        
        for t in range(num_iter):
            # Compute Jacobian
            W = self.compute_jacobian(x, mu)
            
            # Compute precision matrix
            WTW = torch.bmm(W.transpose(1, 2), W)
            I = torch.eye(self.latent_dim).unsqueeze(0).to(mu.device)
            precision = WTW + I
            
            # Compute covariance (inverse of precision)
            covariance = torch.inverse(precision)
            
            # Update mu
            x_recon = self.decode(mu)
            residual = x - x_recon
            
            WT_residual = torch.bmm(W.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)
            mu = torch.bmm(covariance, WT_residual.unsqueeze(-1)).squeeze(-1)
            
        return mu, covariance
    
    def reparameterize_full_cov(self, mu, covariance):
        """Reparameterization with full covariance matrix"""
        batch_size = mu.size(0)
        
        # Cholesky decomposition
        L = torch.linalg.cholesky(covariance)
        
        # Sample
        eps = torch.randn_like(mu)
        z = mu + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)
        
        return z
    
    def forward(self, x):
        # Initial encoding
        mu_init, _ = self.encode(x)
        
        # Iterative mode update
        mu, covariance = self.iterative_mode_update(x, mu_init)
        
        # Sample with full covariance
        z = self.reparameterize_full_cov(mu, covariance)
        
        # Decode
        x_recon = self.decode(z)
        
        return x_recon, mu, covariance, z
    
    def loss_function(self, x, x_recon, mu, covariance, **kwargs):
        """
        Compute ELBO with full covariance Gaussian
        """
        batch_size = x.size(0)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
        
        # KL divergence for full covariance Gaussian
        # KL(N(mu, Sigma) || N(0, I)) = 0.5 * (tr(Sigma) + mu^T mu - k - ln|Sigma|)
        trace = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(-1)
        mu_norm = (mu * mu).sum(-1)
        logdet = torch.logdet(covariance)
        
        kl_loss = 0.5 * torch.mean(trace + mu_norm - self.latent_dim - logdet)
        
        loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample_prior(self, num_samples, device):
        """Sample from standard Gaussian prior"""
        return torch.randn(num_samples, self.latent_dim).to(device)