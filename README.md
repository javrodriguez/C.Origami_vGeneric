### C.Origami generic version with dynamic setup of input features and genomic samples. 

Motivation: Training models using different strategies, i.e., using different number and types of features.

- This version provides a dynamic setup for input features as opposed to the fixed ATAC-seq, ChIP-CTCF and DNA sequence features. To do this, the genomic features are dynamically fetched from the /genomic_features/ directory. Each bigwig file found in that directory will be used as a feature. The bigwig file name (excluding the .bw suffix) provides the name of the feature, which can be used (if required) to set the normalization method for individual features.

- The normalization setup of the genomic features can be customized by making use of two arguments:

   '—-genomic_features_norm': Sets one same normalization type for all the genomic features. Choices=[None, ‘log'], default=None.
  
   '—-feature_norms': Sets the normalization type of specific features using a dictionary-like syntaxis. This argument takes key-value pairs where the key is the feature name and the value is the normalization method selected for that feature. Choices=[None, ‘log'], default=None. If used, the global normalization method set in genomic_features_norm argument will be overriden for the selected features.

- Flexible setup of genomic samples. The argument ‘-—celltype’ now accepts multiple samples which can be set using a comma-separated list of sample names. Example: —celltype MCG0023,MCG0019,MCG0027,MCG0034

- The DNA sequence feature can be excluded by using the ‘—-no-sequence’ flag. 




#### INSTALLATION:
 
1) Download the pipeline repository.
 
git clone https://github.com/javrodriguez/C.Origami_vGeneric.git

cd C.Origami_vGeneric

2) Create conda environment.

source /gpfs/home/rodrij92/home_abl/miniconda3/etc/profile.d/conda.sh
conda create -n corigami_vGeneric python==3.9 pytorch==1.12.0 torchvision==0.13.0 pytorch-cuda=11.8 pandas==1.3.0 matplotlib==3.3.2 pybigwig==0.3.18 omegaconf==2.1.1 tqdm==4.64.0 pytorch-lightning=1.9 scikit-image lightning-bolts mkl==2024.0 -c pytorch -c nvidia

3) Install C.Origami.

conda activate C.Origami_vGeneric

pip install -e .
