import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler


class SyntheticDataset(Dataset):
    """
    Synthetic dataset generator for skewed and heavy-tailed distributions
    """
    
    def __init__(self, n_samples=10000, dim=2, distribution='gaussian', 
                 skewness=0.0, tail_weight=1.0, seed=42):
        """
        Args:
            n_samples: Number of samples
            dim: Dimensionality of data
            distribution: Type of distribution ('gaussian', 'skewed', 'heavy_tail', 'mixed')
            skewness: Skewness parameter (0 = symmetric)
            tail_weight: Heavy tail parameter (higher = heavier tails)
            seed: Random seed
        """
        np.random.seed(seed)
        self.n_samples = n_samples
        self.dim = dim
        self.distribution = distribution
        
        # Generate data based on distribution type
        if distribution == 'gaussian':
            self.data = self._generate_gaussian()
        elif distribution == 'skewed':
            self.data = self._generate_skewed(skewness)
        elif distribution == 'heavy_tail':
            self.data = self._generate_heavy_tail(tail_weight)
        elif distribution == 'mixed':
            self.data = self._generate_mixed(skewness, tail_weight)
        elif distribution == 'asymmetric_laplace':
            self.data = self._generate_asymmetric_laplace(skewness)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Normalize data
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        
    def _generate_gaussian(self):
        """Generate standard Gaussian data"""
        return np.random.randn(self.n_samples, self.dim)
    
    def _generate_skewed(self, skewness):
        """Generate skewed data using skew-normal distribution"""
        data = np.zeros((self.n_samples, self.dim))
        for i in range(self.dim):
            data[:, i] = stats.skewnorm.rvs(a=skewness, size=self.n_samples)
        return data
    
    def _generate_heavy_tail(self, tail_weight):
        """Generate heavy-tailed data using Student-t distribution"""
        df = max(2.1, 10.0 / tail_weight)  # Lower df = heavier tails
        data = np.zeros((self.n_samples, self.dim))
        for i in range(self.dim):
            data[:, i] = stats.t.rvs(df=df, size=self.n_samples)
        return data
    
    def _generate_mixed(self, skewness, tail_weight):
        """Generate mixture of distributions"""
        n_components = 3
        samples_per_component = self.n_samples // n_components
        
        data_list = []
        
        # Component 1: Gaussian
        data1 = np.random.randn(samples_per_component, self.dim)
        data_list.append(data1)
        
        # Component 2: Skewed
        data2 = np.zeros((samples_per_component, self.dim))
        for i in range(self.dim):
            data2[:, i] = stats.skewnorm.rvs(a=skewness, size=samples_per_component)
        data2 += np.array([3, 3])  # Shift mean
        data_list.append(data2)
        
        # Component 3: Heavy-tailed
        remaining = self.n_samples - 2 * samples_per_component
        df = max(2.1, 10.0 / tail_weight)
        data3 = np.zeros((remaining, self.dim))
        for i in range(self.dim):
            data3[:, i] = stats.t.rvs(df=df, size=remaining)
        data3 += np.array([-3, -3])  # Shift mean
        data_list.append(data3)
        
        data = np.vstack(data_list)
        np.random.shuffle(data)
        
        return data
    
    def _generate_asymmetric_laplace(self, skewness):
        """Generate asymmetric Laplace data"""
        data = np.zeros((self.n_samples, self.dim))
        
        for i in range(self.dim):
            # Parameters for asymmetric Laplace
            kappa = np.exp(skewness)  # Asymmetry parameter
            
            # Generate using inverse transform sampling
            u = np.random.uniform(0, 1, self.n_samples)
            
            # Asymmetric Laplace quantile function
            p = kappa / (1 + kappa)
            
            mask = u < p
            data[mask, i] = np.log(u[mask] * (1 + kappa) / kappa)
            data[~mask, i] = -np.log((1 - u[~mask]) * (1 + kappa))
            
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
    
    def get_full_data(self):
        """Get all data as tensor"""
        return torch.FloatTensor(self.data)


def create_data_loaders(config):
    """
    Create data loaders for training and validation
    """
    # Training data
    train_dataset = SyntheticDataset(
        n_samples=config['n_samples_train'],
        dim=config['data_dim'],
        distribution=config['distribution'],
        skewness=config['skewness'],
        tail_weight=config['tail_weight'],
        seed=config['seed']
    )
    
    # Validation data (different seed)
    val_dataset = SyntheticDataset(
        n_samples=config['n_samples_val'],
        dim=config['data_dim'],
        distribution=config['distribution'],
        skewness=config['skewness'],
        tail_weight=config['tail_weight'],
        seed=config['seed'] + 1
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    return train_loader, val_loader


# Additional datasets for benchmarking

class SwissRollDataset(Dataset):
    """Swiss roll dataset for manifold learning"""
    
    def __init__(self, n_samples=10000, noise=0.1):
        from sklearn.datasets import make_swiss_roll
        
        self.data, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
        self.data = self.data[:, [0, 2]]  # Use 2D projection
        
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


class MoonsDataset(Dataset):
    """Two moons dataset"""
    
    def __init__(self, n_samples=10000, noise=0.1):
        from sklearn.datasets import make_moons
        
        self.data, _ = make_moons(n_samples=n_samples, noise=noise)
        
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
