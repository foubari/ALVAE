# File: src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
from tqdm import tqdm
import numpy as np

from models import *
from models.encoders import MLPEncoder, ConvEncoder
from models.decoders import MLPDecoder, ConvDecoder
from datasets import create_data_loaders
from utils import set_seed, create_directories, save_checkpoint, print_model_summary
from evaluate import evaluate_model, visualize_results


def create_model(config):
    """Create VAE model based on configuration"""
    
    # Create encoder
    if config['encoder_type'] == 'mlp':
        encoder = MLPEncoder(
            input_dim=config['data_dim'],
            hidden_dims=config['encoder_hidden_dims'],
            latent_dim=config['latent_dim'],
            activation=config['activation']
        )
    elif config['encoder_type'] == 'conv':
        encoder = ConvEncoder(
            input_channels=config['input_channels'],
            input_size=config['input_size'],
            hidden_dims=config['encoder_hidden_dims'],
            latent_dim=config['latent_dim']
        )
    else:
        raise ValueError(f"Unknown encoder type: {config['encoder_type']}")
    
    # Create decoder
    if config['decoder_type'] == 'mlp':
        decoder = MLPDecoder(
            latent_dim=config['latent_dim'],
            hidden_dims=config['decoder_hidden_dims'],
            output_dim=config['data_dim'],
            activation=config['activation'],
            output_activation=config.get('output_activation', None)
        )
    elif config['decoder_type'] == 'conv':
        decoder = ConvDecoder(
            latent_dim=config['latent_dim'],
            hidden_dims=config['decoder_hidden_dims'],
            output_channels=config['input_channels'],
            output_size=config['input_size']
        )
    else:
        raise ValueError(f"Unknown decoder type: {config['decoder_type']}")
    
    # Create VAE model
    model_name = config['model'].lower()
    
    if model_name == 'gaussian_vae':
        model = GaussianVAE(encoder, decoder, config['latent_dim'], beta=config['beta'])
    elif model_name == 'student_t_vae':
        model = StudentTVAE(encoder, decoder, config['latent_dim'], beta=config['beta'])
    elif model_name == 'vlvae':
        model = VLVAE(encoder, decoder, config['latent_dim'], 
                      num_iterations=config.get('num_iterations', 5), beta=config['beta'])
    elif model_name == 'poisson_vae' or model_name == 'pvae':
        model = PoissonVAE(encoder, decoder, config['latent_dim'], 
                          beta=config['beta'], temperature=config.get('temperature', 1.0))
    elif model_name == 'asymmetric_laplace_vae' or model_name == 'alvae':
        model = AsymmetricLaplaceVAE(encoder, decoder, config['latent_dim'],
                                     beta=config['beta'], mmd_weight=config.get('mmd_weight', 1.0))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def train_epoch(model, train_loader, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    metrics_accumulator = {}
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, data in enumerate(progress_bar):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Compute loss based on model type
        if isinstance(model, GaussianVAE):
            x_recon, mu, logvar, z = outputs
            loss_dict = model.loss_function(data, x_recon, mu, logvar)
        elif isinstance(model, StudentTVAE):
            x_recon, mu, logvar, df, z = outputs
            loss_dict = model.loss_function(data, x_recon, mu, logvar, df)
        elif isinstance(model, VLVAE):
            x_recon, mu, covariance, z = outputs
            loss_dict = model.loss_function(data, x_recon, mu, covariance)
        elif isinstance(model, PoissonVAE):
            x_recon, rates, z = outputs
            loss_dict = model.loss_function(data, x_recon, rates, z)
        elif isinstance(model, AsymmetricLaplaceVAE):
            x_recon, mu, scale, skew, z = outputs
            loss_dict = model.loss_function(data, x_recon, mu, scale, skew, z)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        
        loss = loss_dict['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Accumulate metrics
        train_loss += loss.item()
        for key, value in loss_dict.items():
            if key not in metrics_accumulator:
                metrics_accumulator[key] = 0
            metrics_accumulator[key] += value.item() if torch.is_tensor(value) else value
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Log to tensorboard
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
    
    # Average metrics
    avg_loss = train_loss / len(train_loader)
    avg_metrics = {k: v / len(train_loader) for k, v in metrics_accumulator.items()}
    
    return avg_loss, avg_metrics


def validate(model, val_loader, device):
    """Validate model"""
    model.eval()
    val_loss = 0
    metrics_accumulator = {}
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Compute loss based on model type
            if isinstance(model, GaussianVAE):
                x_recon, mu, logvar, z = outputs
                loss_dict = model.loss_function(data, x_recon, mu, logvar)
            elif isinstance(model, StudentTVAE):
                x_recon, mu, logvar, df, z = outputs
                loss_dict = model.loss_function(data, x_recon, mu, logvar, df)
            elif isinstance(model, VLVAE):
                x_recon, mu, covariance, z = outputs
                loss_dict = model.loss_function(data, x_recon, mu, covariance)
            elif isinstance(model, PoissonVAE):
                x_recon, rates, z = outputs
                loss_dict = model.loss_function(data, x_recon, rates, z)
            elif isinstance(model, AsymmetricLaplaceVAE):
                x_recon, mu, scale, skew, z = outputs
                loss_dict = model.loss_function(data, x_recon, mu, scale, skew, z)
            
            val_loss += loss_dict['loss'].item()
            
            for key, value in loss_dict.items():
                if key not in metrics_accumulator:
                    metrics_accumulator[key] = 0
                metrics_accumulator[key] += value.item() if torch.is_tensor(value) else value
    
    avg_loss = val_loss / len(val_loader)
    avg_metrics = {k: v / len(val_loader) for k, v in metrics_accumulator.items()}
    
    return avg_loss, avg_metrics


def train(config):
    """Main training function"""
    
    # Set random seed
    set_seed(config['seed'])
    
    # Create directories
    model_dir, fig_dir = create_directories(config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    print_model_summary(model)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                          weight_decay=config.get('weight_decay', 0))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Tensorboard writer
    writer = SummaryWriter(os.path.join('logs', config['model']))
    
    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Metrics: {train_metrics}")
        print(f"Val Metrics: {val_metrics}")
        
        # Log to tensorboard
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            
            checkpoint_path = os.path.join(model_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= config.get('early_stopping_patience', 20):
            print(f"Early stopping triggered after {epoch} epochs")
            break
        
        # Periodic visualization
        if epoch % config.get('vis_frequency', 10) == 0:
            save_path = os.path.join(fig_dir, f'epoch_{epoch}.png')
            visualize_results(model, val_loader, device, save_path)
    
    # Final checkpoint
    final_checkpoint_path = os.path.join(model_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, epoch, val_loss, final_checkpoint_path)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    evaluate_model(model, val_loader, device, fig_dir)
    
    writer.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train VAE models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Model name (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.model:
        config['model'] = args.model
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Train model
    model = train(config)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()