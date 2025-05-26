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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

class GenotypeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
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
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

def safe_norm(norm_layer, x):
    # Only apply normalization if batch size > 1
    if x.size(0) > 1:
        return norm_layer(x)
    return x

class ImputationModel(nn.Module):
    def __init__(self, input_size, hidden_size=512, nhead=8, num_layers=6, dropout=0.3):
        super(ImputationModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Enhanced feature embedding
        self.embedding_fc1 = nn.Linear(input_size, hidden_size)
        self.embedding_fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size // 2)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 3)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        # Embedding
        identity = x.squeeze(1)
        x = self.embedding_fc1(x.squeeze(1))
        x = self.norm1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.embedding_fc2(x)
        x = self.norm2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        # No residual here (shapes don't match)
        x = x.unsqueeze(1)
        x = self.pos_encoder(x)
        # Transformer
        transformer_out = self.transformer_encoder(x)
        x = x + transformer_out
        # Attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        # Final classification
        x = x.squeeze(1)
        identity = x
        x = self.fc1(x)
        x = self.norm2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        # Residual only if shapes match
        if x.shape == identity.shape:
            x = x + identity
        identity = x
        x = self.fc2(x)
        x = self.norm3(x)
        x = self.gelu(x)
        x = self.dropout(x)
        if x.shape == identity.shape:
            x = x + identity
        x = self.fc3(x)
        return x

def calculate_ld(variants, max_distance=10000):
    """
    Calculate Linkage Disequilibrium between variants.
    
    Args:
        variants: List of variant objects
        max_distance: Maximum distance between variants to calculate LD
        
    Returns:
        dict: LD values between variant pairs
    """
    ld_values = {}
    for i, var1 in enumerate(variants):
        for j, var2 in enumerate(variants[i+1:], i+1):
            # Only calculate LD for variants within max_distance
            if abs(var1.POS - var2.POS) > max_distance:
                continue
                
            # Calculate D' and r²
            genotypes1 = [g[0] for g in var1.genotypes if g[0] != -1]
            genotypes2 = [g[0] for g in var2.genotypes if g[0] != -1]
            
            if not genotypes1 or not genotypes2:
                continue
                
            # Calculate allele frequencies
            p1 = sum(genotypes1) / (2 * len(genotypes1))
            p2 = sum(genotypes2) / (2 * len(genotypes2))
            
            # Calculate D
            D = 0
            for g1, g2 in zip(genotypes1, genotypes2):
                D += (g1/2 - p1) * (g2/2 - p2)
            D /= len(genotypes1)
            
            # Calculate D'
            Dmax = min(p1*(1-p2), (1-p1)*p2) if D > 0 else min(p1*p2, (1-p1)*(1-p2))
            Dprime = D / Dmax if Dmax != 0 else 0
            
            # Calculate r²
            r2 = (D**2) / (p1*(1-p1)*p2*(1-p2)) if p1*(1-p1)*p2*(1-p2) != 0 else 0
            
            ld_values[(var1.POS, var2.POS)] = {
                'Dprime': Dprime,
                'r2': r2,
                'distance': abs(var1.POS - var2.POS)
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

def extract_features(variants, logger=None):
    if not variants:
        return None, None
    features = []
    labels = []
    all_positions = [v.POS for v in variants]
    min_pos = min(all_positions) if all_positions else 0
    max_pos = max(all_positions) if all_positions else 1
    for variant in variants:
        genotypes = [g[0] for g in variant.genotypes if g[0] != -1]
        if not genotypes:
            continue
        af = sum(genotypes) / (2 * len(genotypes))
        het = sum(1 for g in genotypes if g == 1) / len(genotypes)
        missing_rate = 1 - (len(genotypes) / len(variant.genotypes))
        norm_pos = (variant.POS - min_pos) / (max_pos - min_pos + 1e-8)
        norm_qual = (variant.QUAL if variant.QUAL is not None else 0) / 1000.0
        for sample_idx, genotype in enumerate(variant.genotypes):
            if genotype[0] == -1:
                continue
            feature_vector = [af, het, missing_rate, norm_pos, len(genotypes), norm_qual]
            features.append(feature_vector)
            labels.append(genotype[0])
    features = np.array(features) if features else None
    labels = np.array(labels) if labels else None
    if logger:
        if features is not None and (np.any(np.isnan(features)) or np.any(np.isinf(features))):
            logger.warning(f"[DIAG] NaN/Inf in features. Features sample: {features[:5]}")
        if labels is not None and (np.any(np.isnan(labels)) or np.any(np.isinf(labels))):
            logger.warning(f"[DIAG] NaN/Inf in labels. Labels sample: {labels[:5]}")
        if labels is not None and not np.all(np.isin(labels, [0, 1, 2])):
            logger.warning(f"[DIAG] Invalid label values. Unique labels: {np.unique(labels)} Labels sample: {labels[:10]}")
    return features, labels

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

def process_window(window_tuple, model, optimizer, criterion, device, early_stopping, memory_manager, args, logger=None, previous_model_state=None, imputation_tracker=None):
    chrom, start, end, variants = window_tuple
    window_start_time = time.time()  # Start timing
    
    features, labels = extract_features(variants, logger=logger)
    if features is None or labels is None:
        if logger: logger.warning(f"No valid genotypes for training in window {chrom}:{start}-{end}")
        return None

    # Move data to GPU if available
    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)
    
    # Check if we have enough samples for training
    if len(features) < 2:
        if logger: logger.warning(f"Not enough samples in window {chrom}:{start}-{end} for training")
        return None
    
    # Create dataset and dataloader with multiple workers
    dataset = GenotypeDataset(features.cpu().numpy(), labels.cpu().numpy())
    
    # Adjust batch size if needed
    batch_size = min(args.batch_size, len(dataset))  # Use batch size from arguments
    if batch_size < 2:
        if logger: logger.warning(f"Batch size too small in window {chrom}:{start}-{end}")
        return None
    
    # Configure DataLoader with more robust settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,  # Use num_workers from arguments
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None,
        drop_last=False
    )
    
    # If we have a previous model state, load it
    if previous_model_state is not None:
        model.load_state_dict(previous_model_state)
        if logger: logger.info(f"Loaded previous model state for window {chrom}:{start}-{end}")
    
    # Learning rate scheduler with warmup and cosine annealing
    num_epochs = 50
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,  # Use learning_rate from arguments
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
        anneal_strategy='cos'
    )
    
    model.train()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_targets = []
    
    # Training loop with gradient accumulation and patience
    accumulation_steps = max(1, min(2, len(dataloader) // 2))
    optimizer.zero_grad()
    
    # Early stopping parameters
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    min_delta = 0.001
    best_model_state = None
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_batches = 0
        epoch_preds = []
        epoch_targets = []
        
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch['input'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            
            # Skip batches that are too small
            if inputs.size(0) < 2:
                continue
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            epoch_batches += 1
            
            preds = torch.argmax(outputs, dim=1)
            epoch_preds.extend(preds.cpu().numpy())
            epoch_targets.extend(targets.cpu().numpy())
            memory_manager.check_memory()
        
        # Calculate macro F1 for this epoch
        if epoch_batches > 0:
            macro_f1 = f1_score(epoch_targets, epoch_preds, average='macro', zero_division=0)
        else:
            macro_f1 = 0.0
        
        # Log progress every 5 epochs
        if (epoch + 1) % 5 == 0 and logger:
            logger.info(f"Window {chrom}:{start}-{end} - Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss / max(1, epoch_batches):.4f} - Macro F1: {macro_f1:.4f}")
        
        # Early stopping check
        avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else float('inf')
        if avg_epoch_loss < best_loss - min_delta:
            best_loss = avg_epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            if logger: logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
        # Accumulate for window stats
        all_preds.extend(epoch_preds)
        all_targets.extend(epoch_targets)
        total_loss += epoch_loss
        num_batches += epoch_batches
    
    if num_batches == 0:
        return None
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    early_stopping(avg_loss)
    avg_mem, max_mem, avg_cpu, max_cpu, avg_gpu, max_gpu = memory_manager.get_memory_stats()
    
    # Calculate window processing time
    window_time = time.time() - window_start_time
    
    # Impute missing genotypes for this window
    if imputation_tracker is not None:
        model.eval()
        with torch.no_grad():
            for variant in variants:
                # Always process all variants in the window
                current_genotypes = list(variant.genotypes)
                missing_indices = [i for i, g in enumerate(current_genotypes) if g[0] == -1]
                
                if missing_indices:  # Only process if there are missing genotypes
                    imputed_genotypes = predict_missing_genotypes(model, features, missing_indices, current_genotypes)
                    if imputed_genotypes is not None:
                        # Update only the missing genotypes
                        for idx in missing_indices:
                            current_genotypes[idx] = imputed_genotypes[idx]
                        imputation_tracker.add_imputed_variant(chrom, variant.POS, current_genotypes)
                        if logger:
                            logger.info(f"Imputed {len(missing_indices)} genotypes for variant at {chrom}:{variant.POS}")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix.tolist(),
        'avg_memory_mb': avg_mem,
        'max_memory_mb': max_mem,
        'avg_cpu_percent': avg_cpu,
        'max_cpu_percent': max_cpu,
        'avg_gpu_percent': avg_gpu,
        'max_gpu_percent': max_gpu,
        'num_variants': len(variants),
        'num_batches': num_batches,
        'learning_rate': scheduler.get_last_lr()[0],
        'epochs_trained': epoch + 1,
        'model_state': best_model_state,
        'all_predictions': all_preds,
        'all_targets': all_targets,
        'window_time': window_time,  # Add window processing time
        'chromosome': chrom,  # Add window position information
        'start_pos': start,
        'end_pos': end
    }

def predict_missing_genotypes(model, features, missing_indices, original_genotypes):
    """Predict missing genotypes using the trained model."""
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        # Get probabilities using softmax
        probs = torch.nn.functional.softmax(predictions, dim=1)
        predicted_genotypes = torch.argmax(predictions, dim=1)
        
        # Convert predictions to numpy for easier handling
        predicted_genotypes = predicted_genotypes.cpu().numpy()
        probs = probs.cpu().numpy()
        
        # Create output array with original genotypes (extract only the genotype value)
        output_genotypes = [g[0] if isinstance(g, (list, tuple)) else g for g in original_genotypes]
        
        # Update only missing genotypes
        for idx in missing_indices:
            if idx < len(output_genotypes):  # Ensure index is valid
                output_genotypes[idx] = predicted_genotypes[idx]
        
        # Calculate PL values (phred-scaled likelihoods)
        pl_values = (-10 * np.log10(probs + 1e-10)).astype(np.int32)  # Add small epsilon to avoid log(0)
        
        # For each sample, create a tuple of (genotype, phase, pl_values)
        formatted_genotypes = []
        for i in range(len(output_genotypes)):
            if i in missing_indices:  # Check if this index was missing
                # For imputed genotypes, use the predicted value and calculated PLs
                gt = int(output_genotypes[i])  # Convert to regular int
                # Get phase information from original genotype if available
                phase = original_genotypes[i][2] if len(original_genotypes[i]) > 2 else False
                pls = pl_values[i].tolist()  # Convert to regular list
                formatted_genotypes.append((gt, phase, pls))
            else:
                # For original genotypes, keep the original format
                gt = int(output_genotypes[i])  # Convert to regular int
                phase = original_genotypes[i][2] if len(original_genotypes[i]) > 2 else False
                pls = original_genotypes[i][3:] if len(original_genotypes[i]) > 3 else [0, 0, 0]
                formatted_genotypes.append((gt, phase, pls))
        
        return formatted_genotypes

def create_imputed_vcf(input_vcf, output_vcf, imputation_tracker):
    """Create a new VCF file with imputed genotypes."""
    reader = cyvcf2.Reader(input_vcf)
    writer = cyvcf2.Writer(output_vcf, reader)
    
    total_variants = 0
    total_missing = 0
    total_imputed = 0
    
    for variant in reader:
        total_variants += 1
        missing_count = sum(1 for gt in variant.gt_types if gt == -1)
        total_missing += missing_count
        
        if missing_count > 0:
            # Get imputed genotypes for this variant
            variant_key = f"{variant.CHROM}:{variant.POS}"
            if variant_key in imputation_tracker.imputed_variants:
                imputed_genotypes = imputation_tracker.imputed_variants[variant_key]
                
                # Update genotypes in the variant
                for i, (gt, phase, pls) in enumerate(imputed_genotypes):
                    if variant.gt_types[i] == -1:  # Only update missing genotypes
                        # Update genotype and PL values
                        variant.gt_types[i] = gt
                        variant.gt_phases[i] = phase
                        variant.gt_pls[i] = pls
                        total_imputed += 1
                
                # Update FORMAT fields
                variant.FORMAT = "GT:PL"
                
        writer.write_record(variant)
    
    writer.close()
    
    # Log statistics
    print(f"\nVCF Processing Statistics:")
    print(f"Total variants processed: {total_variants}")
    print(f"Total missing genotypes: {total_missing}")
    print(f"Total genotypes imputed: {total_imputed}")
    print(f"Imputation rate: {(total_imputed/total_missing)*100:.2f}%")

def save_window_info(windows, output_file='window_info.txt'):
    """
    Save detailed information about each window to a file.
    
    Args:
        windows: List of window tuples (chrom, start, end, variants)
        output_file: Path to output file
    """
    print(f"\nSaving window information to {output_file}")
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("Window\tChromosome\tStart\tEnd\tTotal_Variants\tOverlap_Variants\n")
        
        # Process each window
        for i, (chrom, start, end, variants) in enumerate(windows, 1):
            # Get variant positions in current window
            current_positions = set(v.POS for v in variants)
            
            # Calculate overlap with previous window
            overlap_count = 0
            if i > 1:
                prev_chrom, prev_start, prev_end, prev_variants = windows[i-1]
                if chrom == prev_chrom:  # Only calculate overlap for same chromosome
                    prev_positions = set(v.POS for v in prev_variants)
                    overlap_count = len(current_positions.intersection(prev_positions))
            
            # Write window information
            f.write(f"{i}\t{chrom}\t{start:,}\t{end:,}\t{len(variants):,}\t{overlap_count:,}\n")
    
    print(f"Window information saved to {output_file}")

def get_vcf_stats(vcf_path, device, memory_manager):
    """
    Calculate basic statistics and perform imputation.
    
    Args:
        vcf_path (str): Path to the VCF file
        device: Device to use for computation
        memory_manager: Memory manager for large data handling
        
    Returns:
        dict: Dictionary containing statistics and imputation results
    """
    if not os.path.exists(vcf_path):
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")
    
    try:
        print(f"\nProcessing VCF file: {vcf_path}")
        start_time = time.time()
        
        # Create windows
        windows = create_windows(vcf_path)
        
        # Save window information
        save_window_info(windows)
        
        # Process windows sequentially to maintain transfer learning
        window_stats = []
        previous_model = None
        processed_windows = 0
        skipped_windows = 0
        
        for i, window in enumerate(windows, 1):
            print(f"\nProcessing window {i}/{len(windows)}")
            stats = process_window(
                window,
                previous_model,
                device,
                memory_manager
            )
            
            if stats is not None:
                window_stats.append(stats)
                previous_model = None
                processed_windows += 1
            else:
                skipped_windows += 1
        
        print(f"\nProcessed {processed_windows} windows successfully")
        print(f"Skipped {skipped_windows} windows due to errors or insufficient data")
        
        if not window_stats:
            raise ValueError("No windows were successfully processed")
        
        # Create imputed VCF
        output_vcf = vcf_path.replace('.vcf', '_imputed.vcf')
        create_imputed_vcf(vcf_path, output_vcf, None)
        
        # Save metrics for later plotting
        with open('imputation_metrics.json', 'w') as f:
            json.dump(window_stats, f)
        
        # Get number of samples
        reader = cyvcf2.Reader(vcf_path)
        num_samples = len(reader.samples)
        
        # Calculate overall statistics
        total_variants = sum(stat['num_variants'] for stat in window_stats)
        chromosomes = set(stat['window_info']['chromosome'] for stat in window_stats)
        
        total_time = time.time() - start_time
        
        stats = {
            'Number of samples': f"{num_samples:,}",
            'Number of variants': f"{total_variants:,}",
            'Number of chromosomes': f"{len(chromosomes):,}",
            'Number of windows': f"{len(windows):,}",
            'Processed windows': f"{processed_windows:,}",
            'Skipped windows': f"{skipped_windows:,}",
            'Total processing time': f"{total_time:.2f} seconds"
        }
        
        return stats
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        tb = traceback.extract_tb(exc_tb)
        filename, line_no, func, text = tb[-1]
        print(f"\nError in {filename}, line {line_no}, in {func}")
        print(f"Error message: {str(e)}")
        print(f"Code context: {text}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

def plot_metrics(metrics, output_dir='plots'):
    """
    Create comprehensive visualizations of imputation metrics.
    
    Args:
        metrics: List of window metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    losses = [m['loss'] for m in metrics]
    accuracies = [m['accuracy'] for m in metrics]
    f1_scores = [m['f1'] for m in metrics]
    precisions = [m['precision'] for m in metrics]
    recalls = [m['recall'] for m in metrics]
    learning_rates = [m['learning_rate'] for m in metrics]
    memory_usage = [m['avg_memory_mb'] for m in metrics]
    cpu_usage = [m['avg_cpu_percent'] for m in metrics]
    gpu_usage = [m['avg_gpu_percent'] for m in metrics]
    
    # Create figure with subplots
    plt.style.use('bmh')
    fig = plt.figure(figsize=(20, 20))  # Increased figure height
    
    # Loss plot
    ax1 = plt.subplot(4, 2, 1)  # Changed to 4x2 grid
    ax1.plot(losses, label='Loss', color='blue')
    ax1.set_title('Training Loss Over Windows')
    ax1.set_xlabel('Window')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Accuracy metrics
    ax2 = plt.subplot(4, 2, 2)  # Changed to 4x2 grid
    ax2.plot(accuracies, label='Accuracy', color='green')
    ax2.plot(f1_scores, label='F1 Score', color='red')
    ax2.set_title('Accuracy Metrics')
    ax2.set_xlabel('Window')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)
    
    # Precision and Recall
    ax3 = plt.subplot(4, 2, 3)  # Changed to 4x2 grid
    ax3.plot(precisions, label='Precision', color='purple')
    ax3.plot(recalls, label='Recall', color='orange')
    ax3.set_title('Precision and Recall')
    ax3.set_xlabel('Window')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True)
    
    # Learning rate
    ax4 = plt.subplot(4, 2, 4)  # Changed to 4x2 grid
    ax4.plot(learning_rates, color='brown')
    ax4.set_title('Learning Rate')
    ax4.set_xlabel('Window')
    ax4.set_ylabel('Learning Rate')
    ax4.grid(True)
    
    # Memory usage
    ax5 = plt.subplot(4, 2, 5)  # Changed to 4x2 grid
    ax5.plot(memory_usage, color='gray')
    ax5.set_title('Memory Usage')
    ax5.set_xlabel('Window')
    ax5.set_ylabel('Memory (MB)')
    ax5.grid(True)
    
    # CPU usage
    ax6 = plt.subplot(4, 2, 6)  # Changed to 4x2 grid
    ax6.plot(cpu_usage, label='CPU Usage', color='orange')
    ax6.plot(gpu_usage, label='GPU Usage', color='purple')
    ax6.set_title('CPU and GPU Usage')
    ax6.set_xlabel('Window')
    ax6.set_ylabel('Usage (%)')
    ax6.legend()
    ax6.grid(True)
    
    # Confusion matrix
    ax7 = plt.subplot(4, 2, (7, 8))  # Changed to 4x2 grid, spanning last row
    last_cm = np.array(metrics[-1]['confusion_matrix'])
    sns.heatmap(last_cm, annot=True, fmt='d', cmap='Blues', ax=ax7,
                xticklabels=['Homozygous Ref (0)', 'Heterozygous (1)', 'Homozygous Alt (2)'],
                yticklabels=['Homozygous Ref (0)', 'Heterozygous (1)', 'Homozygous Alt (2)'])
    ax7.set_title('Confusion Matrix (Last Window)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/imputation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to JSON for later analysis
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

def safe_process_window(window_info, previous_model=None, logger=None):
    """
    Safely process a window with error handling.
    
    Args:
        window_info: Window information
        previous_model: Previous model for transfer learning
        logger: Logger instance
        
    Returns:
        tuple: (metrics, model) or (None, None) if error
    """
    try:
        return process_window(window_info, previous_model)
    except Exception as e:
        if logger:
            logger.error(f"Error processing window {window_info[0]}:{window_info[1]}-{window_info[2]}: {str(e)}")
        return None, None

def calculate_population_genetics(variants):
    """
    Calculate population genetics statistics for variants.
    
    Args:
        variants: List of variant objects
        
    Returns:
        dict: Population genetics statistics for each variant
    """
    stats = {}
    for variant in variants:
        # Get genotypes, excluding missing values
        genotypes = [g[0] for g in variant.genotypes if g[0] != -1]
        
        if not genotypes:
            stats[variant.POS] = {
                'allele_frequency': 0,
                'heterozygosity': 0,
                'missing_rate': 1.0
            }
            continue
            
        # Calculate allele frequency
        af = sum(genotypes) / (2 * len(genotypes))
        
        # Calculate heterozygosity
        het = sum(1 for g in genotypes if g == 1) / len(genotypes)
        
        # Calculate missing rate
        total_samples = len(variant.genotypes)
        missing_rate = 1 - (len(genotypes) / total_samples)
        
        stats[variant.POS] = {
            'allele_frequency': af,
            'heterozygosity': het,
            'missing_rate': missing_rate
        }
    
    return stats

def optimize_performance():
    """Set up performance optimizations"""
    # Enable GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set number of threads for CPU operations
    torch.set_num_threads(cpu_count())
    
    # Enable cuDNN benchmarking for faster training
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    return device

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class MemoryManager:
    def __init__(self):
        self.temp_files = []
        self.memory_usage = []
        self.cpu_usage = []
        self.gpu_usage = []
    
    def create_mmap_file(self, data):
        """Create a memory-mapped file for large data"""
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(data)
        temp_file.close()
        
        with open(temp_file.name, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        self.temp_files.append((temp_file.name, mm))
        return mm
    
    def check_memory(self):
        """Record current memory usage (RAM), CPU, and GPU usage"""
        import psutil
        import GPUtil
        
        # Record RAM usage
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(mem_mb)
        
        # Record CPU usage
        cpu_percent = process.cpu_percent()
        self.cpu_usage.append(cpu_percent)
        
        # Record GPU usage if available
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_percent = gpus[0].load * 100  # Get usage of first GPU
                    self.gpu_usage.append(gpu_percent)
            except:
                self.gpu_usage.append(0)
        else:
            self.gpu_usage.append(0)
    
    def get_memory_stats(self):
        """Return average and maximum memory, CPU, and GPU usage"""
        if not self.memory_usage:
            return 0, 0, 0, 0, 0, 0
            
        mem_avg = sum(self.memory_usage) / len(self.memory_usage)
        mem_max = max(self.memory_usage)
        
        cpu_avg = sum(self.cpu_usage) / len(self.cpu_usage)
        cpu_max = max(self.cpu_usage)
        
        gpu_avg = sum(self.gpu_usage) / len(self.gpu_usage)
        gpu_max = max(self.gpu_usage)
        
        return mem_avg, mem_max, cpu_avg, cpu_max, gpu_avg, gpu_max
    
    def cleanup(self):
        """Clean up temporary files and memory"""
        for filename, mm in self.temp_files:
            mm.close()
            os.unlink(filename)
        self.temp_files = []
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def checkpoint_model(model, optimizer, epoch, loss, path):
    """Save model checkpoint with automatic cleanup of old checkpoints."""
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Remove previous checkpoint if it exists
        if os.path.exists(path):
            os.remove(path)
            
        # Save new checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)
        
        # Clean up old checkpoints (keep only the last 2)
        checkpoint_dir = os.path.dirname(path)
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_window_*.pt')))
        if len(checkpoints) > 2:
            for old_ckpt in checkpoints[:-2]:
                try:
                    os.remove(old_ckpt)
                except OSError:
                    pass  # Ignore errors during cleanup
                    
    except OSError as e:
        if e.errno == errno.ENOSPC:
            print("WARNING: No space left on device. Skipping checkpoint save.")
        else:
            raise
    except RuntimeError as e:
        if "No space left on device" in str(e):
            print("WARNING: No space left on device. Skipping checkpoint save.")
        else:
            raise

def cleanup_checkpoints(temp_dir='temp'):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def create_windows(vcf_path, window_size=150000, overlap=0.1, min_variants=50, min_samples=2):
    """
    Create sliding windows from VCF file, merging windows with too few samples.
    Args:
        vcf_path (str): Path to the VCF file
        window_size (int): Size of each window in base pairs
        overlap (float): Overlap between windows (0.1 = 10%)
        min_variants (int): Minimum number of variants per window
        min_samples (int): Minimum number of samples per window
    Returns:
        list: List of tuples (chrom, start, end, variants)
    """
    overlap_bp = int(window_size * overlap)  # Calculate overlap in base pairs
    print(f"\nCreating windows of {window_size:,} bp with {overlap*100:.1f}% overlap ({overlap_bp:,} bp)")
    windows = []
    current_window = []
    current_chrom = None
    current_start = None
    current_end = None
    window_variants = 0
    reader = cyvcf2.Reader(vcf_path)
    for variant in reader:
        # Initialize window if this is the first variant
        if current_chrom is None:
            current_chrom = variant.CHROM
            current_start = variant.POS
            current_end = current_start + window_size
        # If we're on a new chromosome, save current window and start new one
        if variant.CHROM != current_chrom:
            if window_variants >= min_variants:
                # Check sample count
                sample_count = sum([sum(1 for g in v.genotypes if g[0] != -1) for v in current_window])
                if sample_count >= min_samples:
                    windows.append((current_chrom, current_start, current_end, current_window))
                else:
                    # Merge with next window (skip for now, will merge below)
                    pass
            current_chrom = variant.CHROM
            current_start = variant.POS
            current_end = current_start + window_size
            current_window = []
            window_variants = 0
        # If variant is beyond current window
        if variant.POS >= current_end:
            if window_variants >= min_variants:
                sample_count = sum([sum(1 for g in v.genotypes if g[0] != -1) for v in current_window])
                if sample_count >= min_samples:
                    windows.append((current_chrom, current_start, current_end, current_window))
                    # Start new window with overlap
                    current_start = int(current_end - overlap_bp)
                    current_end = current_start + window_size
                    current_window = []
                    window_variants = 0
                else:
                    # Merge with next window: don't reset, just extend window
                    current_end = variant.POS + window_size
            else:
                # Extend current window
                current_end = variant.POS + window_size
        # Add variant to current window
        current_window.append(variant)
        window_variants += 1
    # Add the last window if it has enough variants and samples
    if window_variants >= min_variants:
        sample_count = sum([sum(1 for g in v.genotypes if g[0] != -1) for v in current_window])
        if sample_count >= min_samples:
            windows.append((current_chrom, current_start, current_end, current_window))
    print(f"Created {len(windows):,} windows")
    return windows

def generate_reports(metrics, output_dir='reports'):
    """
    Generate comprehensive reports and visualizations for the imputation results.
    Args:
        metrics: List of window metrics
        output_dir: Directory to save reports and plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics
    losses = [m['loss'] for m in metrics]
    accuracies = [m['accuracy'] for m in metrics]
    f1_scores = [m['f1'] for m in metrics]
    precisions = [m['precision'] for m in metrics]
    recalls = [m['recall'] for m in metrics]
    learning_rates = [m['learning_rate'] for m in metrics]
    memory_usage = [m['avg_memory_mb'] for m in metrics]
    max_memory_usage = [m['max_memory_mb'] for m in metrics]
    cpu_usage = [m['avg_cpu_percent'] for m in metrics]
    max_cpu_usage = [m['max_cpu_percent'] for m in metrics]
    gpu_usage = [m['avg_gpu_percent'] for m in metrics]
    max_gpu_usage = [m['max_gpu_percent'] for m in metrics]
    variants_per_window = [m['num_variants'] for m in metrics]
    window_times = [m['window_time'] for m in metrics]
    chromosomes = [m['chromosome'] for m in metrics]
    start_positions = [m['start_pos'] for m in metrics]
    end_positions = [m['end_pos'] for m in metrics]

    # Calculate variant statistics
    avg_variants = np.mean(variants_per_window)
    min_variants = np.min(variants_per_window)
    max_variants = np.max(variants_per_window)
    std_variants = np.std(variants_per_window)

    # Create window information file
    window_info_file = os.path.join(output_dir, 'window_info.csv')
    with open(window_info_file, 'w') as f:
        f.write('Window,Chromosome,Start,End,NumVariants,ProcessingTime\n')
        for i, (chrom, start, end, nvars, time) in enumerate(zip(chromosomes, start_positions, end_positions, variants_per_window, window_times)):
            f.write(f'{i+1},{chrom},{start},{end},{nvars},{time:.2f}\n')

    # Calculate macro and weighted metrics over all windows
    macro_f1 = np.mean(f1_scores)
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_accuracy = np.mean(accuracies)
    weighted_f1 = np.average(f1_scores, weights=variants_per_window)
    weighted_precision = np.average(precisions, weights=variants_per_window)
    weighted_recall = np.average(recalls, weights=variants_per_window)
    weighted_accuracy = np.average(accuracies, weights=variants_per_window)

    # 1. Training Progress Dashboard
    plt.style.use('bmh')
    fig = plt.figure(figsize=(20, 15))
    
    # Loss plot
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(losses, label='Loss', color='blue')
    ax1.set_title('Training Loss Over Windows')
    ax1.set_xlabel('Window')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Accuracy metrics
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(accuracies, label='Accuracy', color='green')
    ax2.plot(f1_scores, label='F1 Score', color='red')
    ax2.set_title('Accuracy Metrics')
    ax2.set_xlabel('Window')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)
    
    # Precision and Recall
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(precisions, label='Precision', color='purple')
    ax3.plot(recalls, label='Recall', color='orange')
    ax3.set_title('Precision and Recall')
    ax3.set_xlabel('Window')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True)
    
    # Learning rate
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(learning_rates, color='brown')
    ax4.set_title('Learning Rate')
    ax4.set_xlabel('Window')
    ax4.set_ylabel('Learning Rate')
    ax4.grid(True)
    
    # Memory usage
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(memory_usage, color='gray', label='Avg Mem')
    ax5.plot(max_memory_usage, color='black', linestyle='--', label='Max Mem')
    ax5.set_title('Memory Usage')
    ax5.set_xlabel('Window')
    ax5.set_ylabel('Memory (MB)')
    ax5.legend()
    ax5.grid(True)
    
    # CPU/GPU usage
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(cpu_usage, label='Avg CPU', color='orange')
    ax6.plot(max_cpu_usage, label='Max CPU', color='red', linestyle='--')
    ax6.plot(gpu_usage, label='Avg GPU', color='purple')
    ax6.plot(max_gpu_usage, label='Max GPU', color='blue', linestyle='--')
    ax6.set_title('CPU and GPU Usage')
    ax6.set_xlabel('Window')
    ax6.set_ylabel('Usage (%)')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Performance Distribution Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    sns.histplot(accuracies, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Accuracy Distribution')
    axes[0, 0].set_xlabel('Accuracy')
    sns.histplot(f1_scores, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('F1 Score Distribution')
    axes[0, 1].set_xlabel('F1 Score')
    sns.histplot(losses, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Loss Distribution')
    axes[1, 0].set_xlabel('Loss')
    sns.histplot(variants_per_window, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Variants per Window Distribution')
    axes[1, 1].set_xlabel('Number of Variants')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Time per window plot
    plt.figure(figsize=(12, 6))
    plt.plot(window_times, marker='o')
    plt.title('Processing Time per Window')
    plt.xlabel('Window')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.savefig(f'{output_dir}/time_per_window.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Window Performance Analysis
    plt.figure(figsize=(15, 6))
    plt.plot(variants_per_window, accuracies, 'o', alpha=0.5)
    plt.title('Accuracy vs. Number of Variants per Window')
    plt.xlabel('Number of Variants')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f'{output_dir}/accuracy_vs_variants.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Variants per Window Plot
    plt.figure(figsize=(15, 6))
    plt.plot(variants_per_window, marker='o')
    plt.axhline(y=avg_variants, color='r', linestyle='--', label=f'Average: {avg_variants:.1f}')
    plt.axhline(y=min_variants, color='g', linestyle='--', label=f'Minimum: {min_variants}')
    plt.axhline(y=max_variants, color='b', linestyle='--', label=f'Maximum: {max_variants}')
    plt.title('Number of Variants per Window')
    plt.xlabel('Window')
    plt.ylabel('Number of Variants')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/variants_per_window.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Generate Detailed Report
    report = f"""Imputation Performance Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Statistics:
-----------------
Total Windows Processed: {len(metrics)}
Total Variants Processed: {sum(variants_per_window):,}

Variant Statistics per Window:
----------------------------
Average Variants: {avg_variants:.1f} ± {std_variants:.1f}
Minimum Variants: {min_variants}
Maximum Variants: {max_variants}

Accuracy Metrics (Macro):
------------------------
F1 Score: {macro_f1:.4f}
Precision: {macro_precision:.4f}
Recall: {macro_recall:.4f}
Accuracy: {macro_accuracy:.4f}

Accuracy Metrics (Weighted):
---------------------------
F1 Score: {weighted_f1:.4f}
Precision: {weighted_precision:.4f}
Recall: {weighted_recall:.4f}
Accuracy: {weighted_accuracy:.4f}

Loss Statistics:
---------------
Average Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}
Minimum Loss: {np.min(losses):.4f}
Maximum Loss: {np.max(losses):.4f}

Resource Usage:
--------------
Average Memory Usage: {np.mean(memory_usage):.2f} MB
Maximum Memory Usage: {np.max(max_memory_usage):.2f} MB
Average CPU Usage: {np.mean(cpu_usage):.2f}%
Maximum CPU Usage: {np.max(max_cpu_usage):.2f}%
Average GPU Usage: {np.mean(gpu_usage):.2f}%
Maximum GPU Usage: {np.max(max_gpu_usage):.2f}%

Processing Time:
---------------
Total Processing Time: {sum(window_times):.2f} seconds
Average Time per Window: {np.mean(window_times):.2f} seconds
Minimum Time per Window: {np.min(window_times):.2f} seconds
Maximum Time per Window: {np.max(window_times):.2f} seconds

Performance Trends:
-----------------
Accuracy Trend: {trend_acc}
F1 Score Trend: {trend_f1}
Loss Trend: {trend_loss}
""".format(
        trend_acc='Improving' if accuracies[-1] > accuracies[0] else 'Declining',
        trend_f1='Improving' if f1_scores[-1] > f1_scores[0] else 'Declining',
        trend_loss='Improving' if losses[-1] < losses[0] else 'Declining'
    )

    # Save report
    with open(f'{output_dir}/imputation_report.txt', 'w') as f:
        f.write(report)

    return report

def main():
    """
    Main function to run the VCF analysis and imputation pipeline.
    """
    # Set up logging
    logger = setup_logger()
    
    # Parse command line arguments first
    parser = argparse.ArgumentParser(description='VCF Analysis and Imputation Pipeline')
    parser.add_argument('vcf_file', help='Path to input VCF file')
    parser.add_argument('--window_size', type=int, default=150000, help='Window size for processing')
    parser.add_argument('--overlap', type=float, default=0.1, help='Overlap between windows (0.1 = 10%)')
    parser.add_argument('--output', help='Path to output VCF file')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for regularization')
    args = parser.parse_args()
    
    try:
        # Verify VCF file exists
        if not os.path.exists(args.vcf_file):
            raise FileNotFoundError(f"VCF file not found: {args.vcf_file}")
        
        # Set default output path if not provided
        if not args.output:
            args.output = args.vcf_file.replace('.vcf', '_imputed.vcf')
        
        # Create windows first
        logger.info(f"Creating windows from {args.vcf_file}...")
        windows = create_windows(
            args.vcf_file,
            window_size=args.window_size,
            overlap=args.overlap,
            min_variants=50
        )
        logger.info(f"Created {len(windows)} windows")
        
        # Initialize imputation tracker with window ranges
        imputation_tracker = ImputationTracker()
        for i, (chrom, start, end, _) in enumerate(windows):
            imputation_tracker.add_window_overlap(chrom, start, end, i)
        
        # Set up device and distributed training
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            num_gpus = torch.cuda.device_count()
            logger.info(f"Using {num_gpus} GPU(s)")
        else:
            device = torch.device('cpu')
            num_gpus = 0
            logger.info("Using CPU")
        
        # Initialize memory manager
        memory_manager = MemoryManager()
        
        # Initialize model and training components
        logger.info("Initializing model and training components...")
        model = ImputationModel(
            input_size=6,
            hidden_size=512,
            nhead=8,
            num_layers=6,
            dropout=0.3
        ).to(device)
        
        if num_gpus > 1:
            model = nn.DataParallel(model)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        early_stopping = EarlyStopping(patience=7)
        
        logger.info("Starting window processing...")
        # Process windows with transfer learning
        total_variants = 0
        total_missing = 0
        total_imputed = 0
        window_stats = []
        failed_windows = []
        previous_model_state = None
        
        # Create initial VCF writer
        reader = cyvcf2.Reader(args.vcf_file)
        num_samples = len(reader.samples)
        writer = cyvcf2.Writer(args.output, reader)
        
        # Keep track of processed variants to avoid duplicates
        processed_variants = set()
        imputed_variants = set()  # Track which variants have been imputed
        
        # First pass to count total missing genotypes
        logger.info("Counting total missing genotypes...")
        for variant in reader:
            total_variants += 1
            missing_count = sum(1 for g in variant.genotypes if g[0] == -1)
            total_missing += missing_count
        reader = cyvcf2.Reader(args.vcf_file)  # Reset reader
        
        # Calculate percentage of missing genotypes
        total_genotypes = total_variants * num_samples
        missing_percentage = (total_missing / total_genotypes) * 100
        
        logger.info(f"Dataset statistics:")
        logger.info(f"Number of samples: {num_samples:,}")
        logger.info(f"Number of variants: {total_variants:,}")
        logger.info(f"Total genotypes: {total_genotypes:,}")
        logger.info(f"Missing genotypes: {total_missing:,} ({missing_percentage:.2f}%)")
        
        for i, window in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Process window with transfer learning and immediate imputation
            stats = process_window(
                window,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                early_stopping=early_stopping,
                memory_manager=memory_manager,
                args=args,  # Pass args to process_window
                logger=logger,
                previous_model_state=previous_model_state,
                imputation_tracker=imputation_tracker
            )
            
            if stats:
                window_stats.append(stats)
                # Save model state for next window
                previous_model_state = stats['model_state']
                logger.info(f"Window {i+1} completed with accuracy: {stats['accuracy']:.4f}, loss: {stats['loss']:.4f}")
                
                # Write imputed variants immediately after processing this window
                chrom, start, end, variants = window
                window_imputed = 0
                for variant in variants:
                    variant_key = (variant.CHROM, variant.POS)
                    if variant_key not in processed_variants:
                        imputed_genotypes = imputation_tracker.get_imputed_genotypes(variant.CHROM, variant.POS)
                        if imputed_genotypes is not None:
                            # Count only newly imputed genotypes
                            original_missing = sum(1 for g in variant.genotypes if g[0] == -1)
                            imputed_missing = sum(1 for g in imputed_genotypes if g[0] == -1)
                            newly_imputed = original_missing - imputed_missing
                            if newly_imputed > 0 and variant_key not in imputed_variants:
                                window_imputed += newly_imputed
                                imputed_variants.add(variant_key)
                            variant.genotypes = imputed_genotypes
                        writer.write_record(variant)
                        processed_variants.add(variant_key)
                
                total_imputed += window_imputed
                
                # Log imputation progress
                if logger:
                    logger.info(f"Window {i+1} imputation stats:")
                    logger.info(f"Variants processed: {len(variants)}")
                    logger.info(f"Newly imputed in this window: {window_imputed}")
                    logger.info(f"Total imputed so far: {total_imputed:,}")
                    logger.info(f"Imputation rate: {(total_imputed/total_missing*100 if total_missing > 0 else 0):.1f}%")
            else:
                failed_windows.append(i)
                logger.warning(f"Window {i+1} failed to process")
            
            memory_manager.check_memory()
            
            # Save checkpoint periodically
            if (i + 1) % 5 == 0:
                checkpoint_path = f'temp/checkpoint_window_{i+1}.pt'
                checkpoint_model(model, optimizer, i+1, stats['loss'] if stats else float('inf'), checkpoint_path)
                logger.info(f"Saved checkpoint at window {i+1}")
        
        # Close the VCF writer
        writer.close()
        
        # Generate reports and visualizations
        if window_stats:
            report = generate_reports(window_stats)
            logger.info("Generated comprehensive reports and visualizations")
            logger.info("\n" + report)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Number of samples: {num_samples:,}")
        logger.info(f"Number of variants: {total_variants:,}")
        logger.info(f"Total genotypes: {total_genotypes:,}")
        logger.info(f"Missing genotypes: {total_missing:,} ({missing_percentage:.2f}%)")
        logger.info(f"Total genotypes imputed: {total_imputed:,}")
        logger.info(f"Final imputation rate: {(total_imputed/total_missing*100 if total_missing > 0 else 0):.1f}%")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        memory_manager.cleanup()
        cleanup_checkpoints('temp')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 