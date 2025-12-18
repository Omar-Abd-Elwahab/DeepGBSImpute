# DeepGBSImpute

DeepGBSImpute: A Reference-Free Transformer-Based Genotype Imputation Framework for Sparse Genotyping-by-Sequencing Data 

## Description

DeepGBSImpute is a tool that uses transformer-based deep learning to impute missing genotypes in VCF files. It processes the data in windows and uses a transformer architecture to learn patterns in the genotype data for accurate imputation.

## Features

- Transformer-based deep learning model for genotype imputation
- Window-based processing of VCF files
- GPU acceleration support
- Comprehensive logging and progress tracking
- Resource usage monitoring
- Visualization of:
  - Resource usage (GPU and CPU)
  - Window statistics (missing genotypes and percentages)
  - Genotype distribution

## Requirements

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Omar-Abd-Elwahab/DeepGBSImpute.git
cd DeepGBSImpute
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python vcf_analyzer.py input.vcf --window_size 150000 --epochs 5 --output_dir output
```

### Arguments

- `vcf_file`: Path to the input VCF file (required)
- `--window_size`: Window size in base pairs (default: 150000)
- `--epochs`: Number of training epochs (default: 5)
- `--output_dir`: Directory for output files (default: 'output')

### Output

The tool generates:
- Imputed VCF file (`imputed.vcf.gz`)
- Log file with detailed progress and metrics
- Resource usage plots
- Window statistics plots
- Genotype distribution plots

## Performance

The tool automatically uses GPU if available, falling back to CPU if not. Performance metrics and resource usage are logged during execution.

## Model Architecture

The pipeline uses a transformer-based deep learning model for genotype imputation:

1. **Input Processing**:
   - Genotype encoding: 0/0 → 0, 0/1 or 1/0 → 1, 1/1 → 2
   - Position normalization: Variant positions normalized to [0,1] range
   - Missing genotype handling: Masked during training, imputed during inference

2. **Transformer Architecture**:
   - Input dimension: 2 (allele states)
   - Hidden dimension: 256
   - Number of transformer layers: 4
   - Number of attention heads: 8
   - Dropout rate: 0.1
   - GELU activation
   - Layer normalization

3. **Training Process**:
   - Window-based processing (default: 150,000 bp)
   - Train/validation/test split (80/10/10)
   - AdamW optimizer
   - Cross-entropy loss
   - Gradient clipping (max_norm=1.0)
   - Early stopping based on validation metrics

4. **Output Layer**:
   - Three-class classification (0/0, 0/1, 1/1 for genotypes)
   - Softmax output
   - Random assignment of 0/1 or 1/0 for heterozygous predictions

5. **Performance Monitoring**:
   - Accuracy, F1 scores (macro and weighted)
   - Precision and recall
   - Confusion matrix
   - Resource usage (GPU memory, CPU utilization)
   - Processing time per phase

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use DeepGBSImpute in your research, please cite:

Abdelwahab, O., Torkamaneh, D. (2026). DeepGBSImpute: A Reference-Free Transformer-Based Genotype Imputation Framework for Sparse Genotyping-By-Sequencing Data. In: Rojas, I., Ortuño, F., Rojas Ruiz, F., Herrera, L.J., Valenzuela, O., Escobar, J.J. (eds) Bioinformatics and Biomedical Engineering. IWBBIO 2025. Lecture Notes in Computer Science(), vol 16050. Springer, Cham. https://doi.org/10.1007/978-3-032-08455-2_1


## Contact

For questions and support, please open an issue on GitHub or contact the maintainer.



