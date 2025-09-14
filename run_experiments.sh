#!/bin/bash

# Script to run all experiments

echo "Starting VAE comparison experiments..."

# Create directories
mkdir -p checkpoints logs figures data/synthetic

# List of models to train
models=("gaussian_vae" "student_t_vae" "vlvae" "pvae" "alvae")

# List of distributions to test
distributions=("gaussian" "skewed" "heavy_tail" "mixed" "asymmetric_laplace")

# Train each model on each distribution
for dist in "${distributions[@]}"; do
    echo "Testing on $dist distribution..."
    
    for model in "${models[@]}"; do
        echo "Training $model on $dist distribution..."
        
        # Update config
        python -c "
import yaml
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['model'] = '$model'
config['distribution'] = '$dist'
with open('configs/config_temp.yaml', 'w') as f:
    yaml.dump(config, f)
"
        
        # Train model
        python src/train.py --config configs/config_temp.yaml
        
        # Evaluate model
        python src/evaluate.py --config configs/config_temp.yaml \
               --checkpoint checkpoints/$model/best_model.pt \
               --output_dir figures/${model}_${dist}
    done
done

# Clean up temp config
rm configs/config_temp.yaml

echo "All experiments completed!"