# DeepFastGBS

A high-throughput genotyping-by-sequencing (GBS) pipeline integrating DeepVariant and GLnexus for accurate variant calling in plant genomics.

## Overview

DeepFastGBS is a comprehensive pipeline designed for efficient and accurate variant calling in plant genomics using Genotyping-by-Sequencing (GBS) data. The pipeline combines the power of DeepVariant for accurate variant calling with GLnexus for joint genotyping, optimized for plant genomes.

## Features

- High-throughput processing of GBS data
- Accurate variant calling using DeepVariant
- Joint genotyping with GLnexus
- Comprehensive quality control metrics
- Detailed performance reports and visualizations
- Memory-efficient processing with window-based analysis
- Support for both CPU and GPU processing
- Detailed logging and error tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Omar-Abd-Elwahab/DeepFastGBS.git
cd DeepFastGBS
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

### Basic Usage

```bash
python vcf_analyzer.py input.vcf.gz --window_size 150000 --overlap 0.1
```

### Command Line Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vcf_file` | Required | Path to input VCF file |
| `--window_size` | 150000 | Size of each window in base pairs |
| `--overlap` | 0.1 | Overlap between windows (0.1 = 10%) |
| `--output` | None | Path to output VCF file (default: input_imputed.vcf) |
| `--batch_size` | 256 | Batch size for training |
| `--num_workers` | 4 | Number of worker processes |
| `--learning_rate` | 0.0005 | Initial learning rate |
| `--weight_decay` | 0.01 | Weight decay for regularization |

### Best Practices

1. **Window Size Selection**:
   - For small genomes (< 100Mb): Use 50,000-100,000 bp windows
   - For medium genomes (100Mb-1Gb): Use 100,000-200,000 bp windows
   - For large genomes (> 1Gb): Use 200,000-500,000 bp windows

2. **Overlap Settings**:
   - Use 10% overlap for most cases
   - Increase overlap to 20% for regions with high variant density
   - Decrease overlap to 5% for regions with low variant density

3. **Batch Size**:
   - Start with 256 for most cases
   - Increase to 512 for high-memory systems
   - Decrease to 128 for low-memory systems

4. **Number of Workers**:
   - Set to number of CPU cores - 1
   - Maximum recommended: 8 workers
   - Minimum recommended: 2 workers

5. **Learning Rate**:
   - Default: 0.0005
   - Range: 0.0001 to 0.001
   - Adjust based on convergence speed

6. **Weight Decay**:
   - Default: 0.01
   - Range: 0.001 to 0.1
   - Increase for overfitting, decrease for underfitting

## Output Files

The pipeline generates several output files in the `reports` directory:

1. `imputation_report.txt`: Comprehensive performance report
2. `window_info.csv`: Detailed information about each window
3. `training_dashboard.png`: Training metrics visualization
4. `performance_distributions.png`: Distribution of performance metrics
5. `time_per_window.png`: Processing time per window
6. `accuracy_vs_variants.png`: Accuracy vs. variant count
7. `variants_per_window.png`: Variant distribution across windows

## Performance Monitoring

The pipeline includes comprehensive performance monitoring:

1. **Memory Usage**:
   - Tracks average and maximum memory usage
   - Monitors CPU and GPU utilization
   - Provides memory optimization recommendations

2. **Processing Time**:
   - Records time per window
   - Tracks total processing time
   - Identifies bottlenecks

3. **Quality Metrics**:
   - Accuracy, F1 score, precision, and recall
   - Variant statistics per window
   - Imputation quality metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use DeepFastGBS in your research, please cite:

```
@software{DeepFastGBS,
  author = {Omar Abd-Elwahab},
  title = {DeepFastGBS: A high-throughput GBS pipeline for plant genomics},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Omar-Abd-Elwahab/DeepFastGBS}
}
```

## Contact

For questions and support, please open an issue on GitHub or contact the maintainer.

## Acknowledgments

- DeepVariant team for the variant calling model
- GLnexus team for the joint genotyping tool
- All contributors and users of the pipeline 