import torch
import torch.nn.functional as F
import numpy as np


def gaussian_nll(x, x_recon, logvar=None):
    """Gaussian negative log-likelihood"""
    if logvar is None:
        return F.mse_loss(x_recon, x, reduction='sum')
    else:
        return 0.5 * torch.sum(logvar + (x - x_recon)**2 / torch.exp(logvar))


def bernoulli_nll(x, x_recon):
    """Bernoulli negative log-likelihood"""
    return F.binary_cross_entropy(x_recon, x, reduction='sum')


def kl_divergence_gaussian(mu, logvar):
    """KL divergence for Gaussian distribution"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def kl_divergence_studentt(mu, logvar, df):
    """Approximate KL divergence for Student-t distribution"""
    # Using Gaussian approximation for tractability
    return kl_divergence_gaussian(mu, logvar)


def kl_divergence_poisson(rates, prior_rate=1.0):
    """KL divergence between Poisson distributions"""
    return torch.sum(rates * torch.log(rates / prior_rate) - (rates - prior_rate))


def mmd_loss(z, prior_samples, kernel='rbf', bandwidths=None):
    """
    Maximum Mean Discrepancy loss
    Args:
        z: Samples from posterior
        prior_samples: Samples from prior
        kernel: Kernel type ('rbf', 'polynomial')
        bandwidths: List of bandwidths for RBF kernel
    """
    if bandwidths is None:
        bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    def rbf_kernel(x, y, bandwidth):
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())
        
        rx = torch.diag(xx).unsqueeze(0).expand_as(xx)
        ry = torch.diag(yy).unsqueeze(0).expand_as(yy)
        
        dxx = rx.t() + rx - 2 * xx
        dyy = ry.t() + ry - 2 * yy
        dxy = rx.t() + ry - 2 * xy
        
        kernel_val = torch.exp(-dxx / (2 * bandwidth**2)).mean()
        kernel_val += torch.exp(-dyy / (2 * bandwidth**2)).mean()
        kernel_val -= 2 * torch.exp(-dxy / (2 * bandwidth**2)).mean()
        
        return kernel_val
    
    def polynomial_kernel(x, y, degree=3, coef=1.0):
        xy = torch.mm(x, y.t())
        return ((xy + coef) ** degree).mean()
    
    if kernel == 'rbf':
        mmd = sum([rbf_kernel(z, prior_samples, bw) for bw in bandwidths])
        return mmd / len(bandwidths)
    elif kernel == 'polynomial':
        return polynomial_kernel(z, prior_samples)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def compute_metrics(x, x_recon, z):
    """Compute additional metrics for monitoring"""
    metrics = {}
    
    # Reconstruction quality
    metrics['mse'] = F.mse_loss(x_recon, x).item()
    metrics['mae'] = F.l1_loss(x_recon, x).item()
    
    # Latent space metrics
    metrics['z_mean'] = z.mean().item()
    metrics['z_std'] = z.std().item()
    metrics['active_units'] = (z.abs() > 0.01).float().mean().item()
    
    # Distribution metrics
    z_flat = z.view(-1)
    metrics['skewness'] = ((z_flat - z_flat.mean()) ** 3).mean() / (z_flat.std() ** 3)
    metrics['kurtosis'] = ((z_flat - z_flat.mean()) ** 4).mean() / (z_flat.var() ** 2)
    
    return metrics


# Model-specific loss functions

def gaussian_vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """Standard Gaussian VAE loss"""
    batch_size = x.size(0)
    recon_loss = gaussian_nll(x, x_recon) / batch_size
    kl_loss = kl_divergence_gaussian(mu, logvar) / batch_size
    
    return {
        'loss': recon_loss + beta * kl_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }


def student_t_vae_loss(x, x_recon, mu, logvar, df, beta=1.0):
    """Student-t VAE loss with robust reconstruction"""
    batch_size = x.size(0)
    
    # Student-t reconstruction loss
    precision = torch.exp(-logvar)
    diff = x - x_recon
    
    recon_loss = -torch.sum(
        torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2) 
        - 0.5 * torch.log(np.pi * df) - 0.5 * logvar
        - (df + 1) / 2 * torch.log(1 + precision * diff**2 / df)
    ) / batch_size
    
    kl_loss = kl_divergence_studentt(mu, logvar, df) / batch_size
    
    return {
        'loss': recon_loss + beta * kl_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }


def vlvae_loss(x, x_recon, mu, covariance, beta=1.0):
    """VLVAE loss with full covariance"""
    batch_size = x.size(0)
    latent_dim = mu.size(1)
    
    recon_loss = gaussian_nll(x, x_recon) / batch_size
    
    # KL with full covariance
    trace = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(-1)
    mu_norm = (mu * mu).sum(-1)
    logdet = torch.logdet(covariance)
    
    kl_loss = 0.5 * torch.mean(trace + mu_norm - latent_dim - logdet)
    
    return {
        'loss': recon_loss + beta * kl_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }


def poisson_vae_loss(x, x_recon, rates, z, beta=1.0, sparsity_weight=0.1):
    """Poisson VAE loss with sparsity"""
    batch_size = x.size(0)
    
    recon_loss = gaussian_nll(x, x_recon) / batch_size
    kl_loss = kl_divergence_poisson(rates) / batch_size
    sparsity_loss = rates.mean()
    
    return {
        'loss': recon_loss + beta * kl_loss + sparsity_weight * sparsity_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'sparsity': sparsity_loss
    }


def alvae_loss(x, x_recon, mu, scale, skew, z, mmd_weight=1.0):
    """Asymmetric Laplace VAE loss with MMD"""
    batch_size = x.size(0)
    device = x.device
    
    recon_loss = gaussian_nll(x, x_recon) / batch_size
    
    # Sample from AL prior for MMD
    prior_mu = torch.zeros_like(z)
    prior_scale = torch.ones_like(z)
    prior_skew = torch.zeros_like(z)
    
    # Simple AL prior sampling
    u = torch.rand_like(z)
    p = torch.sigmoid(prior_skew)
    scale_left = prior_scale / (2 * (1 - p + 1e-8))
    scale_right = prior_scale / (2 * (p + 1e-8))
    
    prior_samples = torch.where(
        u < (1 - p),
        prior_mu + scale_left * torch.log(u / (1 - p + 1e-8)),
        prior_mu - scale_right * torch.log((1 - u) / (p + 1e-8))
    )
    
    mmd = mmd_loss(z, prior_samples)
    
    return {
        'loss': recon_loss + mmd_weight * mmd,
        'recon_loss': recon_loss,
        'mmd_loss': mmd
    }