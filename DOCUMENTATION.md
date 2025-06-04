# DeepGBSImpute Documentation

## Overview
DeepGBSImpute is a deep learning-based genotype imputation tool specifically designed for Genotyping-by-Sequencing (GBS) data. It leverages transformer architecture to capture complex patterns and dependencies in genomic data, enabling accurate imputation of missing genotypes.

## Model Architecture

### Transformer Encoder
The model uses a transformer encoder architecture with the following specifications:
- **Input Layer**: Processes genotype alleles and normalized genomic positions
- **Positional Encoding**: Adds positional information to maintain sequence context
- **Transformer Encoder Layers**: 4 layers, each containing:
  - Multi-head self-attention (8 heads per layer)
  - Feed-forward sublayer
  - Layer normalization
  - Residual connections
- **Output Head**: Projects to 3 genotype classes (0/0, 0/1, 1/1)

### Attention Mechanism
The self-attention mechanism allows the model to:
- Learn relationships between variants within windows
- Capture long-range dependencies in the genome
- Adaptively weight the importance of different variants for imputation

## Data Processing

### Input Requirements
- Multi-sample VCF file containing GBS data
- Unphased genotypes (0/0, 0/1, 1/1)
- Missing genotypes should be encoded as "./."

### Window Creation
- Genomic regions are divided into windows of configurable size
- Default window size: 150,000 base pairs
- Windows are created chromosome by chromosome
- Each window contains multiple variants and their genotypes

### Data Splitting
- Windows are randomly split into:
  - Training set (80%)
  - Test set (20%)
- Split is performed at the window level to prevent data leakage

## Training Process

### Loss Function
- Cross-entropy loss for genotype classification
- Class weights are calculated based on genotype distribution
- Loss is computed only on observed genotypes

### Optimization
- Optimizer: AdamW
- Learning rate: 1e-4
- Gradient clipping: max_norm=1.0

### Training Loop
1. Process each training window
2. Forward pass through transformer
3. Compute loss on observed genotypes
4. Backward pass and optimization
5. Track metrics (accuracy, F1, precision, recall)

## Evaluation Metrics

### Performance Metrics
- Accuracy
- Macro and Weighted F1 scores
- Macro and Weighted Precision
- Macro and Weighted Recall

### Visualization
The tool generates several plots:
1. Test metrics summary
2. Metrics per window
3. Metrics vs missing genotype percentage
4. Resource usage (GPU/CPU)
5. Window statistics
6. Genotype distribution

## Usage

### Command Line Arguments
```bash
python vcf_analyzer.py <vcf_file> [options]

Required arguments:
  vcf_file              Path to input VCF file

Optional arguments:
  --window_size WINDOW_SIZE
                        Window size in base pairs (default: 150000)
  --epochs EPOCHS       Number of training epochs (default: 5)
  --output_dir OUTPUT_DIR
                        Directory for output files (default: 'output')
```

### Output Files
- `imputed.vcf.gz`: Imputed genotypes in VCF format
- `test_metrics.png`: Summary of test set performance
- `test_metrics_per_window.png`: Performance metrics for each window
- `metrics_vs_missing.png`: Relationship between missing data and performance
- `resource_usage.png`: GPU and CPU usage during training
- `window_statistics.png`: Statistics about windows
- `genotype_distribution.png`: Distribution of genotypes
- `imputation.log`: Detailed logging information

## Performance Considerations

### Memory Management
- Automatic batch size calculation based on available memory
- GPU memory clearing after processing each window
- Efficient data loading and processing

### GPU Acceleration
- Automatic GPU detection and utilization
- CUDA support for faster training
- Memory-efficient attention implementation

## Error Handling
- Comprehensive error checking and reporting
- Graceful handling of missing data
- Detailed logging of errors and warnings

## Dependencies
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- cyvcf2
- scikit-learn
- matplotlib
- seaborn
- tqdm
- psutil
- GPUtil

## Citation
If you use DeepGBSImpute in your research, please cite:
```
[Citation information to be added]
```

## License
[License information to be added]

## Contact
[Contact information to be added] 