import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    """Multi-layer perceptron decoder for VAEs"""
    
    def __init__(self, latent_dim, hidden_dims, output_dim, activation='relu', output_activation=None):
        super(MLPDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers
        layers = []
        in_dim = latent_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            elif activation == 'swish':
                layers.append(nn.SiLU())
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim))
        
        # Add output activation if specified
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())
            
        self.decoder = nn.Sequential(*layers)
        
        # For models that need variance output
        self.fc_logvar = None
        
    def forward(self, z):
        """
        Decode latent codes to reconstruction
        Args:
            z: Latent codes [batch_size, latent_dim]
        Returns:
            x_recon: Reconstructed data [batch_size, output_dim]
        """
        return self.decoder(z)
    
    def add_variance_output(self, hidden_dim):
        """Add variance output for probabilistic decoders"""
        self.fc_logvar = nn.Linear(hidden_dim, self.output_dim)
        
    def decode_with_variance(self, z):
        """Decode with mean and variance"""
        h = self.decoder[:-1](z)  # All layers except last
        mu = self.decoder[-1](h)
        
        if self.fc_logvar is not None:
            logvar = self.fc_logvar(h)
            return mu, logvar
        else:
            return mu, None


class ConvDecoder(nn.Module):
    """Convolutional decoder for image generation"""
    
    def __init__(self, latent_dim, hidden_dims, output_channels, output_size):
        super(ConvDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.output_size = output_size
        
        # Initial projection
        self.init_size = output_size // (2 ** len(hidden_dims))
        self.init_channels = hidden_dims[0]
        self.fc = nn.Linear(latent_dim, self.init_channels * self.init_size ** 2)
        
        # Build deconvolutional layers
        modules = []
        in_channels = self.init_channels
        
        for i, h_dim in enumerate(hidden_dims[1:] + [output_channels]):
            is_last = (i == len(hidden_dims) - 1)
            
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim) if not is_last else nn.Identity(),
                    nn.LeakyReLU(0.2) if not is_last else nn.Sigmoid()
                )
            )
            in_channels = h_dim
            
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), self.init_channels, self.init_size, self.init_size)
        return self.decoder(h)


class LinearDecoder(nn.Module):
    """Simple linear decoder for sparse coding models"""
    
    def __init__(self, latent_dim, output_dim, bias=True):
        super(LinearDecoder, self).__init__()
        self.linear = nn.Linear(latent_dim, output_dim, bias=bias)
        
    def forward(self, z):
        return self.linear(z)