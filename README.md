# Asymmetric Laplace Variational AutoEncoder

A comprehensive comparison of ALVAE (Asymmetric Laplace Variational AutoEncoder) against various baseline VAE models for handling skewed and heavy-tailed distributions.

## Models Implemented

1. **Gaussian VAE**: Standard VAE with Gaussian prior and posterior
2. **Student-t VAE**: Robust VAE using Student-t distributions for heavy-tailed data
3. **VLVAE (Variational Laplace VAE)**: VAE with full covariance using Laplace approximation
4. **P-VAE (Poisson VAE)**: VAE with discrete Poisson-distributed latents
5. **ALVAE (Asymmetric Laplace VAE)**: Novel VAE for skewed and heavy-tailed distributions

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
vae-project/
├── src/
│   ├── models/           # VAE model implementations
│   ├── losses.py         # Loss functions
│   ├── datasets.py       # Data generation and loading
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation utilities
│   └── utils.py          # Helper functions
├── configs/              # Configuration files
├── data/                 # Data directory
├── checkpoints/          # Saved models
├── logs/                 # Tensorboard logs
└── figures/              # Generated figures
```

## Usage

### Training a Single Model

```bash
python src/train.py --config configs/config.yaml --model alvae
```

### Training All Baselines

```bash
# Train Gaussian VAE
python src/train.py --config configs/config.yaml --model gaussian_vae

# Train Student-t VAE
python src/train.py --config configs/config.yaml --model student_t_vae

# Train VLVAE
python src/train.py --config configs/config.yaml --model vlvae

# Train Poisson VAE
python src/train.py --config configs/config.yaml --model pvae

# Train Asymmetric Laplace VAE
python src/train.py --config configs/config.yaml --model alvae
```

### Customizing Data Distribution

Edit `configs/config.yaml`:

```yaml
# For Gaussian data
distribution: gaussian
skewness: 0.0
tail_weight: 1.0

# For skewed data
distribution: skewed
skewness: 3.0
tail_weight: 1.0

# For heavy-tailed data
distribution: heavy_tail
skewness: 0.0
tail_weight: 5.0

# For mixed distribution
distribution: mixed
skewness: 2.0
tail_weight: 3.0
```

### Evaluation

```bash
python src/evaluate.py --config configs/config.yaml \
                       --checkpoint checkpoints/alvae/best_model.pt \
                       --output_dir figures/evaluation
```

### Monitoring Training

```bash
tensorboard --logdir logs/
```

## Key Features

- **Modular Design**: Easy to add new VAE variants
- **Flexible Data Generation**: Support for various synthetic distributions
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Tensorboard Integration**: Real-time training monitoring
- **Checkpoint Management**: Automatic model saving and loading

## Results

The models are evaluated on:
- Reconstruction quality (MSE, MAE)
- Distribution matching (Wasserstein distance, KL divergence)
- Latent space properties (active units, correlation, skewness, kurtosis)
- Generation quality (visual comparison, moment matching)

## Adding New Models

To add a new VAE variant:

1. Create a new model class in `src/models/`
2. Inherit from `BaseVAE`
3. Implement required methods
4. Add model creation logic in `src/train.py`
5. Add loss function in `src/losses.py` if needed

## Citation

If you use this code, please cite:

```bibtex
@misc{alvae2025,
  title={ALVAE: A Novel Variational Autoencoder for Skewed and Heavy-Tailed Distributions},
  author={Fouad Oubari, Mohamed El-Baha},
  year={2025}
}
```

## License

MIT License