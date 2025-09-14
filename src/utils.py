import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(config):
    """Create necessary directories"""
    dirs = ['checkpoints', 'logs', 'figures', 'data/synthetic']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Create model-specific directories
    model_dir = os.path.join('checkpoints', config['model'])
    os.makedirs(model_dir, exist_ok=True)
    
    fig_dir = os.path.join('figures', config['model'])
    os.makedirs(fig_dir, exist_ok=True)
    
    return model_dir, fig_dir


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {path}")
    return epoch, loss


def plot_distributions(real_data, generated_data, title="Distribution Comparison", save_path=None):
    """Plot real vs generated distributions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 2D scatter plot
    axes[0].scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, label='Real', s=1)
    axes[0].scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.5, label='Generated', s=1)
    axes[0].set_title('2D Scatter Plot')
    axes[0].legend()
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    
    # Marginal distributions
    for i, ax in enumerate(axes[1:]):
        ax.hist(real_data[:, i], bins=50, alpha=0.5, density=True, label='Real')
        ax.hist(generated_data[:, i], bins=50, alpha=0.5, density=True, label='Generated')
        ax.set_title(f'Marginal Distribution - Dim {i+1}')
        ax.legend()
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_latent_space(model, data_loader, device, save_path=None):
    """Visualize latent space"""
    model.eval()
    latents = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Get latent representation
            if hasattr(model, 'encode'):
                z_params = model.encode(batch)
                if isinstance(z_params, tuple):
                    z = z_params[0]  # Use mean
                else:
                    z = z_params
            else:
                _, _, _, z = model(batch)
            
            latents.append(z.cpu().numpy())
    
    latents = np.vstack(latents)
    
    # Plot latent space
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2D visualization (if latent_dim >= 2)
    if latents.shape[1] >= 2:
        axes[0].scatter(latents[:, 0], latents[:, 1], alpha=0.5, s=1)
        axes[0].set_title('Latent Space (First 2 Dimensions)')
        axes[0].set_xlabel('z1')
        axes[0].set_ylabel('z2')
    
    # Distribution of latent dimensions
    for i in range(min(5, latents.shape[1])):
        axes[1].hist(latents[:, i], bins=50, alpha=0.5, density=True, label=f'z{i+1}')
    axes[1].set_title('Latent Dimension Distributions')
    axes[1].legend()
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def compute_distribution_metrics(real_data, generated_data):
    """Compute distribution comparison metrics"""
    metrics = {}
    
    # Wasserstein distance
    from scipy.stats import wasserstein_distance
    for i in range(real_data.shape[1]):
        metrics[f'wasserstein_dim_{i}'] = wasserstein_distance(
            real_data[:, i], generated_data[:, i]
        )
    
    # KL divergence (using histogram approximation)
    for i in range(real_data.shape[1]):
        hist_real, bins = np.histogram(real_data[:, i], bins=50, density=True)
        hist_gen, _ = np.histogram(generated_data[:, i], bins=bins, density=True)
        
        # Add small epsilon to avoid log(0)
        hist_real = hist_real + 1e-10
        hist_gen = hist_gen + 1e-10
        
        metrics[f'kl_divergence_dim_{i}'] = stats.entropy(hist_real, hist_gen)
    
    # Moments comparison
    metrics['mean_diff'] = np.abs(real_data.mean(0) - generated_data.mean(0)).mean()
    metrics['std_diff'] = np.abs(real_data.std(0) - generated_data.std(0)).mean()
    
    # Skewness and kurtosis
    real_skew = stats.skew(real_data, axis=0)
    gen_skew = stats.skew(generated_data, axis=0)
    metrics['skewness_diff'] = np.abs(real_skew - gen_skew).mean()
    
    real_kurt = stats.kurtosis(real_data, axis=0)
    gen_kurt = stats.kurtosis(generated_data, axis=0)
    metrics['kurtosis_diff'] = np.abs(real_kurt - gen_kurt).mean()
    
    return metrics


def print_model_summary(model):
    """Print model architecture summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*50)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*50)