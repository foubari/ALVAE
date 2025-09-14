import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """Multi-layer perceptron encoder for VAEs"""
    
    def __init__(self, input_dim, hidden_dims, latent_dim, activation='relu'):
        super(MLPEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            elif activation == 'swish':
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            in_dim = h_dim
            
        self.encoder = nn.Sequential(*layers)
        
        # Output layers for mean and log variance (or other parameters)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
        
        # Additional outputs for specific VAE types
        self.fc_skew = None  # For asymmetric distributions
        self.fc_df = None    # For Student-t distribution
        
    def forward(self, x):
        """
        Encode input to latent distribution parameters
        Args:
            x: Input tensor [batch_size, input_dim]
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def add_skew_output(self):
        """Add skew parameter output for asymmetric distributions"""
        in_dim = self.fc_mu.in_features
        self.fc_skew = nn.Linear(in_dim, self.latent_dim)
        
    def add_df_output(self):
        """Add degrees of freedom output for Student-t distribution"""
        in_dim = self.fc_mu.in_features
        self.fc_df = nn.Linear(in_dim, self.latent_dim)
        
    def encode_with_extras(self, x):
        """Encode with additional parameters for special distributions"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        extras = {}
        if self.fc_skew is not None:
            extras['skew'] = self.fc_skew(h)
        if self.fc_df is not None:
            extras['df'] = F.softplus(self.fc_df(h)) + 2.0  # Ensure df > 2
            
        return mu, logvar, extras


class ConvEncoder(nn.Module):
    """Convolutional encoder for image data"""
    
    def __init__(self, input_channels, input_size, hidden_dims, latent_dim):
        super(ConvEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Build convolutional layers
        modules = []
        in_channels = input_channels
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*modules)
        
        # Calculate flattened dimension
        self.flatten_dim = self._get_flatten_dim()
        
        # Output layers
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
    def _get_flatten_dim(self):
        """Calculate the flattened dimension after conv layers"""
        x = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
        x = self.encoder(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar