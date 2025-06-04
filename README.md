### C.Origami generic version with flexible setup of input features. It can be used to train models with different architectures of features.

- The input features are not fixed to the standard features ATAC-seq, ChIP-CTCF and DNA sequence. In this version the genomic features are dynamically fetched from the /genomic_features/ directory. Each bigwig file found in that directory will be used as a feature. The bigwig file name provides the name of the feature.

- The normalization setup of the genomic features can be customized by making use of two arguments:

1) The ‘—genomic_features_norm’ argument sets one same normalization type for all the genomic features. Choices=[None, ‘log'], default=None.

2) The ‘—feature_norms’ argument can set the normalization type of specific features in a dictionary fashion. Key-value pairs where the feature name is the key and the normalization method is the value for that feature. Choices=[None, ‘log'], default=None. 

- Flexible setup of genomic samples. The argument ‘—celltype’ now accepts multiple samples by using a comma-separated list of sample names. Example: —celltype MCG0023,MCG0019,MCG0027,MCG0034

- Using DNA sequence is not a mandatory. By default it requires DNA sequence. Setting a ‘—no-sequence’ flag excludes this feature. 




#### INSTALLATION:
 
1) Download C.Origami_vGeneric repository.
 
git clone https://github.com/javrodriguez/C.Origami_vGen.git

cd C.Origami_vGeneric

2) Create conda environment.

source ./miniconda3/etc/profile.d/conda.sh
conda create -n corigami_vGeneric python==3.9 pytorch==1.12.0 torchvision==0.13.0 pytorch-cuda=11.8 pandas==1.3.0 matplotlib==3.3.2 pybigwig==0.3.18 omegaconf==2.1.1 tqdm==4.64.0 pytorch-lightning=1.9 scikit-image lightning-bolts mkl==2024.0 -c pytorch -c nvidia

3) Install C.Origami.

conda activate C.Origami_vGeneric

pip install -e .
