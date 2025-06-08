### C.Origami Generic Version

A modified version of C.Origami that enables flexible training strategies with dynamic feature setup and multiple genomic samples.

#### Key Features

- **Multiple Genomic Samples**: Train on multiple cell types simultaneously using the `--celltype` argument with comma-separated sample names.
- **Dynamic Genomic Features**: Automatically detect and use any bigwig files in the `/genomic_features/` directory as input features.
- **Flexible Normalization**: Customize feature normalization globally or per-feature:
  - `--genomic_features_norm`: Set uniform normalization for all features (choices: None, 'log')
  - `--feature_norms`: Set feature-specific normalization (e.g., "ctcf:None,atac:log")
- **Optional DNA Sequence**: Exclude DNA sequence input using the `--no-sequence` flag
- **Chromosome-Specific Training**: Train on specific chromosomes using `--test_chromosome`

#### Training

You can train your own model on human or mouse cells using various genomic features.

##### Required Data

1. **Genomic Features**
   - Bigwig files for your features of interest (e.g., ATAC-seq, ChIP-seq, TF activity)
   - Place all feature bigwigs in the `/genomic_features/` directory
   - Feature names are derived from the bigwig filenames (excluding .bw)

2. **Hi-C Data**
   - Experimental Hi-C matrices in npz format
   - One npz file per chromosome
   - Place in the `/hic_matrix/` directory

3. **DNA Sequence** (optional)
   - Compressed FASTA files for each chromosome
   - Place in the `/dna_sequence/` directory
   - Required unless using `--no-sequence`

##### Data Directory Structure

```
root
└── hg38
    ├── centrotelo.bed
    ├── dna_sequence
    │   ├── chr1.fa.gz
    │   ├── chr2.fa.gz
    │   └── ...
    └── cell_type
        ├── genomic_features
        │   ├── feature1.bw
        │   ├── feature2.bw
        │   └── ...
        └── hic_matrix
            ├── chr1.npz
            ├── chr2.npz
            └── ...
```

##### Training Command

```bash
corigami-train [options]
```

###### Key Options

- **Data and Run Directories**
  - `--data-root`: Root path of training data (required)
  - `--assembly`: Genome assembly (default: hg38)
  - `--celltype`: Comma-separated list of cell types
  - `--save_path`: Path for model checkpoints (default: checkpoints)

- **Training Parameters**
  - `--max-epochs`: Maximum training epochs (default: 80)
  - `--patience`: Early stopping patience (default: 80)
  - `--batch-size`: Batch size (default: 8)
  - `--num-gpu`: Number of GPUs to use (default: 4)
  - `--num-workers`: DataLoader workers (default: 16)

- **Feature Configuration**
  - `--genomic_features_norm`: Global feature normalization (None/log)
  - `--feature_norms`: Feature-specific normalization (e.g., "ctcf:None,atac:log")
  - `--no-sequence`: Disable DNA sequence input
  - `--test_chromosome`: Train on specific chromosome (default: chr20 if used without value)

##### Example Training Strategies

1. **Standard Training with One Sample**
   ```bash
   corigami-train --data-root ./data --celltype MCG0023 --feature_norms atac:'log',ctcf:None 
   ```
2. **Standard Training with One Sample (no DNA input for CTCF)**
   ```bash
   corigami-train --data-root ./data --celltype MCG0023 --genomic_features_norm 'log'
   ```
      
3. **Standard Training with Multiple Samples**
   ```bash
   corigami-train --data-root ./data --celltype MCG0023,MCG0019 --feature_norms atac:'log',ctcf:None 
   ```

4. **Training with Custom Feature Normalization (e.g. large number of predicted transcription factor activities and ATAC-seq)**
   ```bash
   corigami-train --data-root ./data --celltype MCG0023 --genomic_features_norm None --feature_norms atac:'log'
   ```

5. **Standard training without DNA Sequence**
   ```bash
   corigami-train --data-root ./data --celltype MCG0023 --feature_norms atac:'log',ctcf:None  --no-sequence
   ```

6. **Training on Specific Chromosome**
   ```bash
   corigami-train --data-root ./data --celltype MCG0023 --test_chromosome chr1
   ```

#### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/javrodriguez/C.Origami_vGeneric.git
   cd C.Origami_vGeneric
   ```

2. Create conda environment:
   ```bash
   conda create -n corigami_vGeneric python==3.9 pytorch==1.12.0 torchvision==0.13.0 pytorch-cuda=11.8 pandas==1.3.0 matplotlib==3.3.2 pybigwig==0.3.18 omegaconf==2.1.1 tqdm==4.64.0 pytorch-lightning=1.9 scikit-image lightning-bolts mkl==2024.0 -c pytorch -c nvidia
   ```

3. Install C.Origami:
   ```bash
   conda activate corigami_vGeneric
   pip install .
   ```

#### Usage with Singularity (no installation required)

   Alternatively, C.Origami can be used without need for installation by using a singularity image.
   We provide images for maxATAC, C.Origami and C.Origami dependencies (useful when working with a dev version of C.Origami).
   All of those are available at https://doi.org/10.5281/zenodo.15604832.
   Check the ./singularity directory for example usage scripts and the .def files used to generate those images.
