# File: src/evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from utils import plot_distributions, plot_latent_space, compute_distribution_metrics


def evaluate_model(model, data_loader, device, save_dir=None):
    """Comprehensive model evaluation"""
    model.eval()
    
    all_data = []
    all_recon = []
    all_latent = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Extract reconstruction and latent
            x_recon = outputs[0]
            z = outputs[-1]
            
            all_data.append(batch.cpu().numpy())
            all_recon.append(x_recon.cpu().numpy())
            all_latent.append(z.cpu().numpy())
    
    all_data = np.vstack(all_data)
    all_recon = np.vstack(all_recon)
    all_latent = np.vstack(all_latent)
    
    # Generate samples from prior
    num_samples = len(all_data)
    generated = model.sample(num_samples, device).cpu().numpy()
    
    # Compute metrics
    metrics = compute_distribution_metrics(all_data, generated)
    recon_metrics = compute_distribution_metrics(all_data, all_recon)
    
    print("\nGeneration Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nReconstruction Metrics:")
    for key, value in recon_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualizations
    if save_dir:
        # Plot distributions
        plot_distributions(
            all_data, generated,
            title=f"{model.__class__.__name__} - Generated vs Real",
            save_path=os.path.join(save_dir, 'generated_vs_real.png')
        )
        
        plot_distributions(
            all_data, all_recon,
            title=f"{model.__class__.__name__} - Reconstructed vs Real",
            save_path=os.path.join(save_dir, 'reconstructed_vs_real.png')
        )
        
        # Plot latent space
        plot_latent_analysis(all_latent, save_path=os.path.join(save_dir, 'latent_analysis.png'))
        
        # Plot loss landscape (for small models)
        if all_latent.shape[1] == 2:
            plot_loss_landscape(model, data_loader, device, 
                               save_path=os.path.join(save_dir, 'loss_landscape.png'))
    
    return metrics


def plot_latent_analysis(latent, save_path=None):
    """Analyze latent space properties"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Distribution of each latent dimension
    n_dims = min(5, latent.shape[1])
    for i in range(n_dims):
        axes[0, 0].hist(latent[:, i], bins=50, alpha=0.5, density=True, label=f'z{i+1}')
    axes[0, 0].set_title('Latent Distributions')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    
    # Correlation matrix
    corr = np.corrcoef(latent.T)
    sns.heatmap(corr, cmap='coolwarm', center=0, ax=axes[0, 1])
    axes[0, 1].set_title('Latent Correlation Matrix')
    
    # Active units
    active_units = (np.abs(latent) > 0.01).mean(axis=0)
    axes[0, 2].bar(range(len(active_units)), active_units)
    axes[0, 2].set_title('Active Units')
    axes[0, 2].set_xlabel('Latent Dimension')
    axes[0, 2].set_ylabel('Activity Rate')
    
    # Skewness and kurtosis
    skewness = stats.skew(latent, axis=0)
    kurtosis = stats.kurtosis(latent, axis=0)
    
    axes[1, 0].bar(range(len(skewness)), skewness)
    axes[1, 0].set_title('Skewness per Dimension')
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('Skewness')
    
    axes[1, 1].bar(range(len(kurtosis)), kurtosis)
    axes[1, 1].set_title('Kurtosis per Dimension')
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('Kurtosis')
    
    # 2D visualization if applicable
    if latent.shape[1] >= 2:
        axes[1, 2].scatter(latent[:, 0], latent[:, 1], alpha=0.5, s=1)
        axes[1, 2].set_title('Latent Space (First 2 Dims)')
        axes[1, 2].set_xlabel('z1')
        axes[1, 2].set_ylabel('z2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_loss_landscape(model, data_loader, device, save_path=None):
    """Plot loss landscape for 2D latent space"""
    model.eval()
    
    # Get a batch of data
    data = next(iter(data_loader)).to(device)
    
    # Create grid
    z_range = np.linspace(-3, 3, 50)
    Z1, Z2 = np.meshgrid(z_range, z_range)
    
    loss_landscape = np.zeros_like(Z1)
    
    with torch.no_grad():
        for i in range(Z1.shape[0]):
            for j in range(Z1.shape[1]):
                z = torch.tensor([[Z1[i, j], Z2[i, j]]], dtype=torch.float32).to(device)
                x_recon = model.decode(z)
                
                # Compute reconstruction loss
                loss = torch.nn.functional.mse_loss(x_recon, data[:1]).item()
                loss_landscape[i, j] = loss
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(Z1, Z2, loss_landscape, levels=20, cmap='viridis')
    plt.colorbar(label='Reconstruction Loss')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title('Loss Landscape')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_results(model, data_loader, device, save_path=None):
    """Quick visualization during training"""
    model.eval()
    
    with torch.no_grad():
        # Get a batch
        data = next(iter(data_loader)).to(device)
        
        # Forward pass
        outputs = model(data)
        x_recon = outputs[0]
        
        # Generate samples
        samples = model.sample(len(data), device)
        
        # Convert to numpy
        data_np = data.cpu().numpy()
        recon_np = x_recon.cpu().numpy()
        samples_np = samples.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot if 2D
    if data_np.shape[1] >= 2:
        axes[0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.5, s=10)
        axes[0].set_title('Original Data')
        
        axes[1].scatter(recon_np[:, 0], recon_np[:, 1], alpha=0.5, s=10)
        axes[1].set_title('Reconstructed')
        
        axes[2].scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=10)
        axes[2].set_title('Generated Samples')
    else:
        axes[0].hist(data_np[:, 0], bins=50, alpha=0.7)
        axes[0].set_title('Original Data')
        
        axes[1].hist(recon_np[:, 0], bins=50, alpha=0.7)
        axes[1].set_title('Reconstructed')
        
        axes[2].hist(samples_np[:, 0], bins=50, alpha=0.7)
        axes[2].set_title('Generated Samples')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # Example usage for standalone evaluation
    import argparse
    import yaml
    from train import create_model
    from datasets import create_data_loaders
    
    parser = argparse.ArgumentParser(description='Evaluate trained VAE model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='figures/evaluation', 
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and load checkpoint
    model = create_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data loader
    _, val_loader = create_data_loaders(config)
    
    # Evaluate
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_model(model, val_loader, device, args.output_dir)