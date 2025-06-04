import cyvcf2
import numpy as np
from collections import Counter
import sys
import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import psutil
import GPUtil
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import gc
import mmap
import tempfile
import traceback
import argparse
from scipy.stats import mode
import shutil
import glob
import errno
import copy
import warnings
import torch.nn.functional as F
import math
import torch.optim as optim
import torch.cuda.amp as amp
import gzip
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.module')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.core.fromnumeric')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas.core.dtypes.common')

# Set PyTorch to use deterministic algorithms if needed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set numpy to use a fixed random seed
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class GenotypeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return {
            'input': feature,
            'target': label
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer('pe', self._get_pe(max_len, d_model))

    def _get_pe(self, length, d_model):
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            # Expand positional encoding if needed
            new_pe = self._get_pe(seq_len, self.d_model).to(x.device)
            self.pe = new_pe
        return x + self.pe[:seq_len, :].unsqueeze(0)

def safe_norm(norm_layer, x):
    # Only apply normalization if batch size > 1
    if x.size(0) > 1:
        return norm_layer(x)
    return x

class ImputationModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_dim
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim + 1)  # +1 for position
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3 possible genotypes: 0/0, 0/1 or 1/0, 1/1
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, positions=None):
        # Add position information
        if positions is not None:
            x = torch.cat([x, positions], dim=-1)
        
        # Input normalization and projection
        x = self.input_norm(x)
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer layers
        x = self.transformer(x)
        
        # Output head
        x = self.output_head(x)
        
        return x

class GenotypeLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, pred, target):
        # Ensure target is the right shape for cross entropy
        if len(target.shape) == 1:
            target = target.long()  # Ensure target is long type
        
        # Calculate loss with class weights if provided
        if self.class_weights is not None:
            loss = F.cross_entropy(pred, target, weight=self.class_weights)
        else:
            loss = F.cross_entropy(pred, target)
        return loss

def calculate_class_weights(genotype_distribution):
    """Calculate class weights based on genotype distribution."""
    # Extract counts from the unphased genotype distribution
    counts = genotype_distribution['unphased']
    
    # Initialize all possible genotypes with a small count
    all_genotypes = {'0/0': 1, '0/1': 1, '1/0': 1, '1/1': 1}
    
    # Update with actual counts from the data
    for genotype, count in counts.items():
        if genotype != './.':  # Skip missing genotypes
            all_genotypes[genotype] = max(count, 1)  # Ensure at least count of 1
    
    # Calculate total count of valid genotypes
    total = sum(all_genotypes.values())
    
    # Calculate weights with a softer balancing approach
    # Use square root of inverse frequency to reduce extreme weights
    weights = {genotype: np.sqrt(total / (count * len(all_genotypes))) 
              for genotype, count in all_genotypes.items()}
    
    # Convert to tensor
    weight_tensor = torch.tensor([weights[g] for g in ['0/0', '0/1', '1/0', '1/1']], 
                               dtype=torch.float32)
    
    # Normalize weights to have mean 1.0
    weight_tensor = weight_tensor / weight_tensor.mean()
    
    return weight_tensor

def calculate_ld(variants):
    """
    Calculate linkage disequilibrium (LD) between variants using r².
    
    Args:
        variants: List of variant objects containing genotype information
        
    Returns:
        Dictionary mapping variant position pairs to LD information
    """
    ld_values = {}
    n_variants = len(variants)
    
    # For each pair of variants
    for i in range(n_variants):
        for j in range(i + 1, n_variants):
            var1 = variants[i]
            var2 = variants[j]
            
            # Skip if variants are too far apart (e.g., > 1Mb)
            if abs(var1.POS - var2.POS) > 1_000_000:
                continue
                
            # Extract allele frequencies
            p1 = sum(1 for gt in var1.genotypes if gt[0] == 1 or gt[1] == 1) / (2 * len(var1.genotypes))
            p2 = sum(1 for gt in var2.genotypes if gt[0] == 1 or gt[1] == 1) / (2 * len(var2.genotypes))
            
            # Calculate D (linkage disequilibrium coefficient)
            D = 0
            n_samples = len(var1.genotypes)
            for k in range(n_samples):
                gt1 = var1.genotypes[k]
                gt2 = var2.genotypes[k]
                
                # Skip if either genotype is missing
                if gt1[0] == -1 or gt2[0] == -1:
                    continue
                
                # Count alleles
                a1 = gt1[0] + gt1[1]  # Number of alternate alleles for variant 1
                a2 = gt2[0] + gt2[1]  # Number of alternate alleles for variant 2
                
                # Update D
                D += (a1/2 - p1) * (a2/2 - p2)
            
            D /= n_samples
            
            # Calculate r²
            r2 = (D * D) / (p1 * (1-p1) * p2 * (1-p2)) if p1 * (1-p1) * p2 * (1-p2) > 0 else 0
            
            # Store LD information
            ld_values[(var1.POS, var2.POS)] = {
                'r2': r2,
                'D': D,
                'p1': p1,
                'p2': p2
            }
    
    return ld_values

def setup_logger(log_file='imputation_diagnostics.log'):
    """Set up logging with both file and console handlers."""
    logger = logging.getLogger('imputation')
    logger.setLevel(logging.INFO)
    
    # File handler for detailed logging
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler for important messages only
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)  # Only show warnings and errors in console
    
    # Create formatters and add them to the handlers
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    fh.setFormatter(file_formatter)
    ch.setFormatter(console_formatter)
    
    # Add the handlers to the logger
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

def extract_features(variants):
    """Extract features and genotype class labels from a window of variants."""
    num_variants = len(variants)
    num_samples = len(variants[0].genotypes)
    
    # Initialize tensors
    features = torch.zeros((num_samples, num_variants, 2))  # [samples, variants, alleles]
    positions = torch.zeros((num_variants,))  # [variants]
    missing_mask = torch.zeros((num_samples, num_variants), dtype=torch.bool)  # [samples, variants]
    labels = torch.zeros((num_samples, num_variants), dtype=torch.long)  # [samples, variants]
    
    # Process each variant
    for i, variant in enumerate(variants):
        positions[i] = variant.POS
        for j, genotype in enumerate(variant.genotypes):
            if genotype[0] != -1:  # Not missing
                features[j, i, 0] = genotype[0]
                features[j, i, 1] = genotype[1]
                # Convert alleles to class: 0/0=0, 0/1 or 1/0=1, 1/1=2
                if genotype[0] == 0 and genotype[1] == 0:
                    labels[j, i] = 0  # homozygous reference
                elif (genotype[0] == 0 and genotype[1] == 1) or (genotype[0] == 1 and genotype[1] == 0):
                    labels[j, i] = 1  # heterozygous
                elif genotype[0] == 1 and genotype[1] == 1:
                    labels[j, i] = 2  # homozygous alternate
                else:
                    labels[j, i] = 0  # fallback
            else:
                missing_mask[j, i] = True
                labels[j, i] = -1  # Mark missing as -1
    
    # Normalize positions to [0, 1] range
    if positions.numel() > 0:
        min_pos = positions.min()
        max_pos = positions.max()
        if max_pos > min_pos:
            positions = (positions - min_pos) / (max_pos - min_pos)
    
    # Add small noise to positions to prevent exact duplicates
    positions = positions + torch.randn_like(positions) * 1e-6
    
    # Reshape positions to match features shape
    positions = positions.unsqueeze(0).unsqueeze(-1)  # [1, variants, 1]
    positions = positions.expand(num_samples, -1, -1)  # [samples, variants, 1]
    
    return features, positions, labels, missing_mask

class ImputationTracker:
    def __init__(self):
        self.imputed_variants = {}  # {(chrom, pos): imputed_genotypes}
        self.window_ranges = []     # [(chrom, start, end, window_id)]
        self.processed_variants = set()  # Track which variants have been processed
        self.imputation_stats = {
            'total_variants': 0,
            'total_missing': 0,
            'total_imputed': 0
        }
    
    def add_window_overlap(self, chrom, start, end, window_id):
        """Track which windows overlap with each variant position"""
        self.window_ranges.append((chrom, start, end, window_id))
    
    def should_impute(self, chrom, pos):
        """Check if a variant should be imputed in this window"""
        variant_key = (chrom, pos)
        if variant_key in self.processed_variants:
            return False
        self.processed_variants.add(variant_key)
        return True
    
    def add_imputed_variant(self, chrom, pos, genotypes):
        """Store imputed genotypes for a variant"""
        # Verify that all genotypes are valid (not -1)
        if any(g[0] == -1 for g in genotypes):
            return False
        
        self.imputed_variants[(chrom, pos)] = genotypes
        self.imputation_stats['total_variants'] += 1
        self.imputation_stats['total_imputed'] += sum(1 for g in genotypes if g[0] != -1)
        return True
    
    def get_imputed_genotypes(self, chrom, pos):
        """Get previously imputed genotypes for a variant"""
        return self.imputed_variants.get((chrom, pos))
    
    def get_stats(self):
        """Get imputation statistics"""
        return self.imputation_stats

def calculate_optimal_batch_size(available_memory, num_samples=10):
    """Calculate optimal batch size based on available memory."""
    # Estimate memory per sample (in MB)
    memory_per_sample = 0.1  # Conservative estimate
    
    # Calculate maximum batch size that fits in memory
    max_batch_size = int(available_memory * 0.8 / memory_per_sample)  # Use 80% of available memory
    
    # Ensure batch size is at least 1 and at most 32
    return max(1, min(32, max_batch_size))

def split_train_test(variants, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Split known genotypes into training, validation and test sets while maintaining class balance.
    Ensures no data leakage between sets by splitting at the variant level.
    
    Args:
        variants: List of variant objects
        train_ratio: Ratio of known genotypes to use for training
        val_ratio: Ratio of known genotypes to use for validation
        test_ratio: Ratio of known genotypes to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_indices, val_indices, test_indices, test_true_values)
    """
    np.random.seed(random_seed)
    
    # First, split variants into train/val/test sets
    n_variants = len(variants)
    indices = np.arange(n_variants)
    np.random.shuffle(indices)
    
    n_train = int(n_variants * train_ratio)
    n_val = int(n_variants * val_ratio)
    
    train_variant_indices = indices[:n_train]
    val_variant_indices = indices[n_train:n_train + n_val]
    test_variant_indices = indices[n_train + n_val:]
    
    # Now collect genotypes for each set
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Process training variants
    for var_idx in train_variant_indices:
        variant = variants[var_idx]
        for sample_idx, gt in enumerate(variant.genotypes):
            if gt[0] != -1:  # Known genotype
                train_indices.append((var_idx, sample_idx))
    
    # Process validation variants
    for var_idx in val_variant_indices:
        variant = variants[var_idx]
        for sample_idx, gt in enumerate(variant.genotypes):
            if gt[0] != -1:  # Known genotype
                val_indices.append((var_idx, sample_idx))
    
    # Process test variants
    for var_idx in test_variant_indices:
        variant = variants[var_idx]
        for sample_idx, gt in enumerate(variant.genotypes):
            if gt[0] != -1:  # Known genotype
                test_indices.append((var_idx, sample_idx))
    
    # Store true values for test set
    test_true_values = []
    for var_idx, sample_idx in test_indices:
        test_true_values.append(variants[var_idx].genotypes[sample_idx])
    
    return train_indices, val_indices, test_indices, test_true_values

class CosineAnnealingWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warmup learning rate scheduler."""
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super(CosineAnnealingWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return [self.min_lr + (base_lr - self.min_lr) * 
                   (1 + math.cos(math.pi * progress)) / 2 
                   for base_lr in self.base_lrs]

def get_device():
    """Get the best available device with proper initialization."""
    if torch.cuda.is_available():
        try:
            # Initialize CUDA
            torch.cuda.init()
            # Get device
            device = torch.device('cuda')
            # Test CUDA
            torch.cuda.empty_cache()
            return device
        except Exception as e:
            print(f"CUDA initialization failed: {e}. Falling back to CPU.")
            return torch.device('cpu')
    return torch.device('cpu')

def calculate_metrics(predictions, targets, logger=None):
    """Calculate comprehensive metrics for model evaluation."""
    # Convert predictions and targets to numpy arrays if they are PyTorch tensors
    if torch.is_tensor(predictions):
        pred_np = predictions.cpu().numpy()
    else:
        pred_np = predictions
        
    if torch.is_tensor(targets):
        target_np = targets.cpu().numpy()
    else:
        target_np = targets
    
    # Handle empty arrays
    if len(pred_np) == 0 or len(target_np) == 0:
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'macro_precision': 0.0,
            'weighted_precision': 0.0,
            'macro_recall': 0.0,
            'weighted_recall': 0.0,
            'confusion_matrix': np.zeros((3, 3))  # Now 3x3 instead of 4x4
        }
    
    # Define all possible labels (0/0, 0/1 or 1/0, 1/1)
    all_labels = np.array([0, 1, 2])
    
    # Calculate metrics
    try:
        metrics = {
            'accuracy': accuracy_score(target_np, pred_np),
            'macro_f1': f1_score(target_np, pred_np, average='macro', zero_division=0, labels=all_labels),
            'weighted_f1': f1_score(target_np, pred_np, average='weighted', zero_division=0, labels=all_labels),
            'macro_precision': precision_score(target_np, pred_np, average='macro', zero_division=0, labels=all_labels),
            'weighted_precision': precision_score(target_np, pred_np, average='weighted', zero_division=0, labels=all_labels),
            'macro_recall': recall_score(target_np, pred_np, average='macro', zero_division=0, labels=all_labels),
            'weighted_recall': recall_score(target_np, pred_np, average='weighted', zero_division=0, labels=all_labels)
        }
        
        # Add confusion matrix with all labels
        cm = confusion_matrix(target_np, pred_np, labels=all_labels)
        metrics['confusion_matrix'] = cm
        
        return metrics
    except Exception as e:
        if logger:
            logger.error(f"Error calculating metrics: {str(e)}")
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'macro_precision': 0.0,
            'weighted_precision': 0.0,
            'macro_recall': 0.0,
            'weighted_recall': 0.0,
            'confusion_matrix': np.zeros((3, 3))  # Now 3x3 instead of 4x4
        }

def process_window(window, model, device, is_training=True, logger=None, optimizer=None):
    """Process a single window for training or imputation."""
    try:
        chrom, start, end, variants = window
        
        # Skip empty windows
        if not variants:
            return {'loss': 0.0, 'accuracy': 0.0, 'macro_f1': 0.0, 'confusion_matrix': np.zeros((3, 3))}
        
        # Extract features and labels
        features, positions, labels, missing_mask = extract_features(variants)
        
        # Move data to device
        features = features.to(device)
        positions = positions.to(device)
        labels = labels.to(device)
        missing_mask = missing_mask.to(device)
        
        if is_training:
            model.train()
            
            # Forward pass
            outputs = model(features, positions)
            
            # Calculate loss only on non-missing labels
            valid_mask = (labels != -1)
            if valid_mask.any():
                # Apply log_softmax for numerical stability
                log_probs = F.log_softmax(outputs, dim=-1)
                
                # Calculate loss
                loss = F.nll_loss(
                    log_probs.view(-1, 3)[valid_mask.view(-1)],
                    labels.view(-1)[valid_mask.view(-1)]
                )
                
                # Backward pass
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    predictions = torch.argmax(outputs, dim=-1)
                    metrics = calculate_metrics(predictions.view(-1), labels.view(-1), logger)
                    metrics['loss'] = loss.item()
                
                return metrics
            else:
                return {'loss': 0.0, 'accuracy': 0.0, 'macro_f1': 0.0, 'confusion_matrix': np.zeros((3, 3))}
        else:
            model.eval()
            with torch.no_grad():
                # Forward pass
                outputs = model(features, positions)
                probs = F.softmax(outputs, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
                
                # Calculate metrics only on known genotypes
                known_mask = ~missing_mask
                metrics = calculate_metrics(
                    predictions[known_mask].view(-1),
                    labels[known_mask].view(-1),
                    logger
                )
                metrics['loss'] = F.cross_entropy(outputs.view(-1, 3), labels.view(-1), ignore_index=-1).item()
                
                # Impute ALL missing genotypes
                imputed_genotypes = []
                for i, variant in enumerate(variants):
                    variant_genotypes = []
                    for j in range(len(variant.genotypes)):
                        if missing_mask[j, i]:  # If genotype is missing
                            pred = predictions[j, i].item()
                            # Convert prediction to genotype
                            if pred == 0:
                                variant_genotypes.append((0, 0))  # 0/0
                            elif pred == 1:
                                variant_genotypes.append((0, 1))  # 0/1
                            elif pred == 2:
                                variant_genotypes.append((1, 1))  # 1/1
                        else:
                            variant_genotypes.append(variant.genotypes[j])
                    imputed_genotypes.append(variant_genotypes)
                
                return metrics, imputed_genotypes
    except Exception as e:
        if logger:
            logger.error(f"Error processing window: {str(e)}")
        return {'loss': 0.0, 'accuracy': 0.0, 'macro_f1': 0.0, 'confusion_matrix': np.zeros((3, 3))}

def create_windows(vcf_file, window_size=150000):
    """Create windows from VCF file for transformer-based imputation."""
    print(f"\nCreating windows of size {window_size:,} bp")
    
    # Initialize windows
    windows = []
    current_chrom = None
    current_window = []
    window_start = None
    window_end = None
    
    # Read variants
    reader = cyvcf2.Reader(vcf_file)
    for variant in reader:
        # Skip variants with no genotype information
        if variant.gt_types is None:
            continue
            
        if current_chrom is None:
            current_chrom = variant.CHROM
            window_start = variant.POS
            window_end = window_start + window_size
        
        # If we're on a new chromosome or past the window end, save current window and start new one
        if variant.CHROM != current_chrom or variant.POS >= window_end:
            if current_window:  # Save window if it has variants
                windows.append((current_chrom, window_start, window_end, current_window))
            
            # Start new window
            current_chrom = variant.CHROM
            window_start = variant.POS
            window_end = window_start + window_size
            current_window = []
        
        # Add variant if it has any genotype information
        if len(variant.gt_types) > 0:
            current_window.append(variant)
    
    # Save the last window
    if current_window:
        windows.append((current_chrom, window_start, window_end, current_window))
    
    reader.close()
    
    # Split windows into train/test sets (80/20 split)
    np.random.seed(42)
    np.random.shuffle(windows)
    split_idx = int(len(windows) * 0.8)
    train_windows = windows[:split_idx]
    test_windows = windows[split_idx:]
    
    print(f"\nWindow Statistics:")
    print(f"Total windows: {len(windows)}")
    print(f"Training windows: {len(train_windows)}")
    print(f"Test windows: {len(test_windows)}")
    
    return train_windows, test_windows

def plot_resource_usage(resource_history, output_dir):
    """Plot GPU and CPU usage over time."""
    plt.figure(figsize=(15, 5))
    
    # Plot GPU memory usage
    plt.subplot(1, 2, 1)
    plt.plot(resource_history['gpu_memory'], label='GPU Memory (MB)')
    plt.title('GPU Memory Usage')
    plt.xlabel('Time Step')
    plt.ylabel('Memory (MB)')
    plt.legend()
    
    # Plot CPU usage
    plt.subplot(1, 2, 2)
    plt.plot(resource_history['cpu_percent'], label='CPU Usage (%)')
    plt.title('CPU Usage')
    plt.xlabel('Time Step')
    plt.ylabel('Usage (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resource_usage.png'))
    plt.close()

def plot_window_statistics(window_stats, output_dir):
    """Plot statistics about windows."""
    plt.figure(figsize=(15, 5))
    
    # Plot missing genotypes per window
    plt.subplot(1, 2, 1)
    plt.hist(window_stats['missing_per_window'], bins=50)
    plt.title('Distribution of Missing Genotypes per Window')
    plt.xlabel('Number of Missing Genotypes')
    plt.ylabel('Frequency')
    
    # Plot missing percentage per window
    plt.subplot(1, 2, 2)
    plt.hist(window_stats['missing_percentage'], bins=50)
    plt.title('Distribution of Missing Percentage per Window')
    plt.xlabel('Missing Percentage')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'window_statistics.png'))
    plt.close()

def plot_genotype_distribution(genotype_counts, output_dir):
    """Plot distribution of genotypes."""
    plt.figure(figsize=(10, 5))
    genotypes = ['0/0', '0/1', '1/1']
    counts = [genotype_counts[gt] for gt in genotypes]
    plt.bar(genotypes, counts)
    plt.title('Distribution of Genotypes')
    plt.xlabel('Genotype')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'genotype_distribution.png'))
    plt.close()

def monitor_resources():
    """Monitor GPU and CPU usage."""
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]  # Get first GPU
        gpu_memory = gpu.memoryUsed
    
    cpu_percent = psutil.cpu_percent()
    return {'gpu_memory': gpu_memory, 'cpu_percent': cpu_percent}

def collect_window_statistics(window):
    """Collect statistics for a single window."""
    chrom, start, end, variants = window
    num_variants = len(variants)
    missing_count = sum(1 for v in variants for gt in v.genotypes if gt[0] == -1)
    total_genotypes = sum(len(v.genotypes) for v in variants)
    missing_percentage = (missing_count / total_genotypes * 100) if total_genotypes > 0 else 0
    
    return {
        'variants_per_window': num_variants,
        'missing_per_window': missing_count,
        'window_sizes': end - start,
        'missing_percentage': missing_percentage
    }

def plot_test_metrics(test_metrics, output_dir):
    """Plot all test metrics in a single bar plot."""
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1', 
              'Macro Precision', 'Weighted Precision',
              'Macro Recall', 'Weighted Recall']
    values = [test_metrics['accuracy'],
             test_metrics['macro_f1'],
             test_metrics['weighted_f1'],
             test_metrics['macro_precision'],
             test_metrics['weighted_precision'],
             test_metrics['macro_recall'],
             test_metrics['weighted_recall']]
    
    # Create color map
    colors = ['#2ecc71', '#3498db', '#2980b9',  # Greens and blues for F1
             '#e74c3c', '#c0392b',  # Reds for Precision
             '#f1c40f', '#f39c12']  # Yellows for Recall
    
    # Create bar plot
    bars = plt.bar(metrics, values, color=colors)
    
    # Customize plot
    plt.title('Test Metrics Summary', pad=20)
    plt.ylabel('Score')
    plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'test_metrics.png'))
    plt.close()

def plot_test_metrics_per_window(test_metrics_list, output_dir):
    """Plot test metrics (macro and weighted) for each window."""
    plt.figure(figsize=(15, 10))
    
    # Extract metrics for each window
    windows = range(len(test_metrics_list))
    macro_f1 = [m['macro_f1'] for m in test_metrics_list]
    weighted_f1 = [m['weighted_f1'] for m in test_metrics_list]
    macro_precision = [m['macro_precision'] for m in test_metrics_list]
    weighted_precision = [m['weighted_precision'] for m in test_metrics_list]
    macro_recall = [m['macro_recall'] for m in test_metrics_list]
    weighted_recall = [m['weighted_recall'] for m in test_metrics_list]
    
    # Create subplots
    plt.subplot(3, 1, 1)
    plt.plot(windows, macro_f1, 'b-', label='Macro F1')
    plt.plot(windows, weighted_f1, 'r-', label='Weighted F1')
    plt.title('F1 Scores per Window')
    plt.xlabel('Window Index')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(windows, macro_precision, 'b-', label='Macro Precision')
    plt.plot(windows, weighted_precision, 'r-', label='Weighted Precision')
    plt.title('Precision Scores per Window')
    plt.xlabel('Window Index')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(windows, macro_recall, 'b-', label='Macro Recall')
    plt.plot(windows, weighted_recall, 'r-', label='Weighted Recall')
    plt.title('Recall Scores per Window')
    plt.xlabel('Window Index')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_metrics_per_window.png'))
    plt.close()

def plot_metrics_vs_missing(test_metrics_list, window_stats, output_dir):
    """Plot macro and weighted metrics against missing genotype percentage for each window."""
    try:
        # Get missing percentages for test windows
        missing_percentages = window_stats['missing_percentage']
        
        # Ensure we use the same number of windows for both metrics and missing percentages
        n_windows = min(len(missing_percentages), len(test_metrics_list))
        missing_percentages = missing_percentages[:n_windows]
        
        # Extract metrics for each window
        macro_f1 = [m['macro_f1'] for m in test_metrics_list[:n_windows]]
        weighted_f1 = [m['weighted_f1'] for m in test_metrics_list[:n_windows]]
        macro_precision = [m['macro_precision'] for m in test_metrics_list[:n_windows]]
        weighted_precision = [m['weighted_precision'] for m in test_metrics_list[:n_windows]]
        macro_recall = [m['macro_recall'] for m in test_metrics_list[:n_windows]]
        weighted_recall = [m['weighted_recall'] for m in test_metrics_list[:n_windows]]
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        plt.subplot(3, 1, 1)
        plt.scatter(missing_percentages, macro_f1, c='blue', label='Macro F1', alpha=0.6)
        plt.scatter(missing_percentages, weighted_f1, c='red', label='Weighted F1', alpha=0.6)
        plt.title('F1 Scores vs Missing Genotype Percentage')
        plt.xlabel('Missing Genotype Percentage')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.scatter(missing_percentages, macro_precision, c='blue', label='Macro Precision', alpha=0.6)
        plt.scatter(missing_percentages, weighted_precision, c='red', label='Weighted Precision', alpha=0.6)
        plt.title('Precision Scores vs Missing Genotype Percentage')
        plt.xlabel('Missing Genotype Percentage')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.scatter(missing_percentages, macro_recall, c='blue', label='Macro Recall', alpha=0.6)
        plt.scatter(missing_percentages, weighted_recall, c='red', label='Weighted Recall', alpha=0.6)
        plt.title('Recall Scores vs Missing Genotype Percentage')
        plt.xlabel('Missing Genotype Percentage')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_vs_missing.png'))
        plt.close()
        
    except Exception as e:
        print(f"\nError in plot_metrics_vs_missing: {str(e)}")
        print(f"Number of windows in missing_percentages: {len(window_stats['missing_percentage'])}")
        print(f"Number of windows in test_metrics_list: {len(test_metrics_list)}")
        # Create an empty plot with error message
        plt.figure(figsize=(15, 10))
        plt.text(0.5, 0.5, f"Error plotting metrics: {str(e)}", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.savefig(os.path.join(output_dir, 'metrics_vs_missing.png'))
        plt.close()

def main():
    """Main function to run the imputation pipeline."""
    # Start timing the entire pipeline
    pipeline_start_time = time.time()
    
    parser = argparse.ArgumentParser(description='VCF Analysis and Imputation Pipeline')
    parser.add_argument('vcf_file', help='Path to the input VCF file')
    parser.add_argument('--window_size', type=int, default=150000, help='Window size in base pairs')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory for output files')
    args = parser.parse_args()
    
    # Print default values and configuration
    print("\nConfiguration:")
    print("=" * 50)
    print(f"Input VCF file: {args.vcf_file}")
    print(f"Window size: {args.window_size:,} bp")
    print(f"Number of epochs: {args.epochs}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logging
    log_file = os.path.join(args.output_dir, 'imputation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('imputation')

    # Report VCF statistics
    logger.info(f"\nAnalyzing input VCF file: {args.vcf_file}")
    reader = cyvcf2.Reader(args.vcf_file)
    
    # Count variants and samples
    num_variants = 0
    num_samples = len(reader.samples)
    missing_genotypes = 0
    total_genotypes = 0
    
    logger.info("Counting variants and missing genotypes...")
    for variant in tqdm(reader, desc="Reading VCF"):
        num_variants += 1
        for gt in variant.genotypes:
            total_genotypes += 1
            if gt[0] == -1:  # Missing genotype
                missing_genotypes += 1
    
    # Calculate statistics
    missing_percentage = (missing_genotypes / total_genotypes) * 100 if total_genotypes > 0 else 0
    
    # Report statistics
    logger.info("\nVCF Statistics:")
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Number of variants: {num_variants}")
    logger.info(f"Total genotypes: {total_genotypes}")
    logger.info(f"Missing genotypes: {missing_genotypes} ({missing_percentage:.2f}%)")
    
    # Close and reopen reader for further processing
    reader.close()
    
    try:
        # Initialize tracking dictionaries
        resource_history = {'gpu_memory': [], 'cpu_percent': []}
        window_stats = {
            'variants_per_window': [],
            'missing_per_window': [],
            'window_sizes': [],
            'missing_percentage': []
        }
        genotype_counts = {'0/0': 0, '0/1': 0, '1/1': 0}
        
        # Collect initial genotype distribution
        reader = cyvcf2.Reader(args.vcf_file)
        for variant in reader:
            for gt in variant.genotypes:
                if gt[0] != -1:
                    if gt[0] == 0 and gt[1] == 0:
                        genotype_counts['0/0'] += 1
                    elif (gt[0] == 0 and gt[1] == 1) or (gt[0] == 1 and gt[1] == 0):
                        genotype_counts['0/1'] += 1
                    elif gt[0] == 1 and gt[1] == 1:
                        genotype_counts['1/1'] += 1
        reader.close()
        
        # Create windows
        logger.info(f"\nCreating windows from {args.vcf_file}...")
        train_windows, test_windows = create_windows(
            args.vcf_file,
            window_size=args.window_size
        )
        
        if not train_windows or not test_windows:
            raise ValueError("No valid windows created from the VCF file")
        
        # Initialize model and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ImputationModel(input_dim=2, hidden_dim=256, num_layers=4, num_heads=8).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # Initialize metrics tracking
        metrics_history = {
            'train': [],
            'val': [],
            'test': []
        }
        
        # Track best metrics
        best_metrics = {
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'loss': float('inf')
        }
        
        # Initialize timing dictionaries
        timing_stats = {
            'training': [],
            'validation': [],
            'testing': [],
            'imputation': 0.0
        }
        
        # Collect window statistics
        for window in train_windows + test_windows:
            stats = collect_window_statistics(window)
            for key in window_stats:
                window_stats[key].append(stats[key])
        
        # Training loop
        logger.info("Starting training...")
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            
            model.train()
            epoch_train_metrics = []
            
            # Training phase with tqdm
            train_pbar = tqdm(train_windows, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
            for window in train_pbar:
                metrics = process_window(window, model, device, is_training=True, logger=logger, optimizer=optimizer)
                epoch_train_metrics.append(metrics)
                
                # Update progress bar with current metrics
                train_pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'macro_f1': f"{metrics['macro_f1']:.4f}",
                    'weighted_f1': f"{metrics['weighted_f1']:.4f}"
                })
                
                # Clear GPU memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average training metrics
            avg_train_metrics = {
                k: np.mean([m[k] for m in epoch_train_metrics]) 
                for k in epoch_train_metrics[0].keys() 
                if k != 'confusion_matrix'
            }
            metrics_history['train'].append(avg_train_metrics)
            
            # Monitor resources
            resources = monitor_resources()
            resource_history['gpu_memory'].append(resources['gpu_memory'])
            resource_history['cpu_percent'].append(resources['cpu_percent'])
            
            # Validation phase with tqdm
            val_start_time = time.time()
            model.eval()
            epoch_val_metrics = []
            val_pbar = tqdm(train_windows, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
            for window in val_pbar:
                metrics, _ = process_window(window, model, device, is_training=False, logger=logger)
                epoch_val_metrics.append(metrics)
                
                # Update progress bar with current metrics
                val_pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'macro_f1': f"{metrics['macro_f1']:.4f}",
                    'weighted_f1': f"{metrics['weighted_f1']:.4f}"
                })
                
                # Clear GPU memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average validation metrics
            avg_val_metrics = {
                k: np.mean([m[k] for m in epoch_val_metrics]) 
                for k in epoch_val_metrics[0].keys() 
                if k != 'confusion_matrix'
            }
            metrics_history['val'].append(avg_val_metrics)
            
            # Update best metrics
            if avg_val_metrics['macro_f1'] > best_metrics['macro_f1']:
                best_metrics['macro_f1'] = avg_val_metrics['macro_f1']
            if avg_val_metrics['weighted_f1'] > best_metrics['weighted_f1']:
                best_metrics['weighted_f1'] = avg_val_metrics['weighted_f1']
            if avg_val_metrics['loss'] < best_metrics['loss']:
                best_metrics['loss'] = avg_val_metrics['loss']
            
            # Calculate timing for this epoch
            epoch_time = time.time() - epoch_start_time
            val_time = time.time() - val_start_time
            train_time = epoch_time - val_time
            
            timing_stats['training'].append(train_time)
            timing_stats['validation'].append(val_time)
            
            # Log epoch summary with best metrics and timing
            logger.info(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
            logger.info("Training Metrics:")
            for k, v in avg_train_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
            logger.info("Validation Metrics:")
            for k, v in avg_val_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
            logger.info("Best Metrics So Far:")
            for k, v in best_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
            logger.info(f"Timing:")
            logger.info(f"  Training time: {train_time:.2f} seconds")
            logger.info(f"  Validation time: {val_time:.2f} seconds")
            logger.info(f"  Total epoch time: {epoch_time:.2f} seconds")
        
        # Test phase with tqdm
        logger.info("\nEvaluating on test set...")
        test_start_time = time.time()
        model.eval()
        test_metrics = []
        
        test_pbar = tqdm(test_windows, desc='Test Phase')
        for window in test_pbar:
            metrics, _ = process_window(window, model, device, is_training=False, logger=logger)
            test_metrics.append(metrics)
            
            # Update progress bar with current metrics
            test_pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'macro_f1': f"{metrics['macro_f1']:.4f}",
                'weighted_f1': f"{metrics['weighted_f1']:.4f}"
            })
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calculate average test metrics
        avg_test_metrics = {
            k: np.mean([m[k] for m in test_metrics]) 
            for k in test_metrics[0].keys() 
            if k != 'confusion_matrix'
        }
        
        # Add confusion matrix to test metrics
        avg_test_metrics['confusion_matrix'] = np.sum([m['confusion_matrix'] for m in test_metrics], axis=0)
        metrics_history['test'] = avg_test_metrics
        
        # Record test timing
        test_time = time.time() - test_start_time
        timing_stats['testing'].append(test_time)
        logger.info(f"\nTest Phase Summary:")
        logger.info(f"Test time: {test_time:.2f} seconds")
        for k, v in avg_test_metrics.items():
            if k != 'confusion_matrix':
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}:")
                logger.info(f"    {v}")

        # Generate all plots
        plot_resource_usage(resource_history, args.output_dir)
        plot_window_statistics(window_stats, args.output_dir)
        plot_genotype_distribution(genotype_counts, args.output_dir)
        plot_test_metrics(avg_test_metrics, args.output_dir)
        plot_test_metrics_per_window(test_metrics, args.output_dir)
        plot_metrics_vs_missing(test_metrics, window_stats, args.output_dir)
        
        # Write imputed VCF using windows
        output_vcf = os.path.join(args.output_dir, 'imputed.vcf.gz')
        logger.info(f"\nWriting imputed genotypes to {output_vcf}")

        # Start timing imputation
        imputation_start_time = time.time()

        # Read header once
        header_lines = []
        with gzip.open(args.vcf_file, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    header_lines.append(line)
                else:
                    break

        # Write header
        with gzip.open(output_vcf, 'wt') as out_vcf:
            out_vcf.writelines(header_lines)

        # Process and write variants in windows
        logger.info("Processing and writing variants in windows...")
        
        # Create windows for the entire VCF
        all_windows = []
        current_chrom = None
        current_window = []
        window_start = None
        window_end = None
        
        reader = cyvcf2.Reader(args.vcf_file)
        for variant in reader:
            if variant.gt_types is None:
                continue
                
            if current_chrom is None:
                current_chrom = variant.CHROM
                window_start = variant.POS
                window_end = window_start + args.window_size
            
            # If we're on a new chromosome or past the window end, save current window and start new one
            if variant.CHROM != current_chrom or variant.POS >= window_end:
                if current_window:  # Save window if it has variants
                    all_windows.append((current_chrom, window_start, window_end, current_window))
                
                # Start new window
                current_chrom = variant.CHROM
                window_start = variant.POS
                window_end = window_start + args.window_size
                current_window = []
            
            # Add variant if it has any genotype information
            if len(variant.gt_types) > 0:
                current_window.append(variant)
        
        # Save the last window
        if current_window:
            all_windows.append((current_chrom, window_start, window_end, current_window))
        
        reader.close()

        # Process and write each window
        with gzip.open(output_vcf, 'at') as out_vcf:  # 'at' for append text mode
            for window in tqdm(all_windows, desc="Processing windows"):
                chrom, start, end, variants = window
                
                # Skip if no variants in window
                if not variants:
                    continue
                
                # Process window
                features, positions, _, _ = extract_features(variants)
                features = features.to(device)
                positions = positions.to(device)
                
                with torch.no_grad():
                    outputs = model(features, positions)
                    predictions = torch.argmax(outputs, dim=-1)
                
                # Write variants in this window
                for i, variant in enumerate(variants):
                    gt_strs = []
                    for j, gt in enumerate(variant.genotypes):
                        if gt[0] == -1:  # Missing genotype
                            pred = predictions[j, i].item()
                            if pred == 0:
                                gt_strs.append('0/0')
                            elif pred == 1:
                                gt_strs.append('0/1')
                            elif pred == 2:
                                gt_strs.append('1/1')
                        else:
                            gt_strs.append(f'{gt[0]}/{gt[1]}')
                    
                    vcf_line = [
                        variant.CHROM,
                        str(variant.POS),
                        variant.ID if variant.ID else '.',
                        variant.REF,
                        ','.join(variant.ALT),
                        str(variant.QUAL) if variant.QUAL is not None else '.',
                        variant.FILTER if variant.FILTER else '.',
                        variant.INFO.get('INFO', '.'),
                        'GT',
                        '\t'.join(gt_strs)
                    ]
                    out_vcf.write('\t'.join(vcf_line) + '\n')
                
                # Clear GPU memory after each window
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Record imputation timing
        timing_stats['imputation'] = time.time() - imputation_start_time
        logger.info(f"VCF writing completed in {timing_stats['imputation']:.2f} seconds")
        
        # Calculate and report total pipeline timing
        total_time = time.time() - pipeline_start_time
        avg_train_time = np.mean(timing_stats['training'])
        avg_val_time = np.mean(timing_stats['validation'])
        avg_test_time = np.mean(timing_stats['testing'])
        
        logger.info("\nPipeline Timing Summary:")
        logger.info(f"Total pipeline time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info(f"Average training time per epoch: {avg_train_time:.2f} seconds")
        logger.info(f"Average validation time per epoch: {avg_val_time:.2f} seconds")
        logger.info(f"Average test time: {avg_test_time:.2f} seconds")
        logger.info(f"Imputation time: {timing_stats['imputation']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
