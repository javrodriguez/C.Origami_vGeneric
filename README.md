### C.Origami generic version with dynamic setup of input features and genomic samples. 

Motivation: Build a modified C.Origami version that allows the setup of different training strategies, i.e., using different number/types of features.



#### - Training using multiple genomic samples / celltypes.

The argument '--celltype' now accepts multiple samples. This can be set using a comma-separated list of sample names. 

#### - Dynamic setup for genomic features.

In this version the genomic features are dynamically fetched from the /genomic_features/ directory, as opposed to the fixed ATAC-seq, ChIP-CTCF and DNA sequence features used in the original C.Origami. Each bigwig file found in the /genomic_features/ directory will be used as a feature. The bigwig file name (excluding the .bw suffix) provides the name of the feature, which can be used (if required) to set the normalization method for individual features (see next).

The normalization setup can be customized by making use of the following two arguments:

'--genomic_features_norm': Sets one same normalization type for all the genomic features. Choices=[None, 'log'], default=None.

'--feature_norms': Sets the normalization type of specific features using a dictionary-like syntaxis. This argument takes key-value pairs where the key is the feature name and the value is the normalization method selected for that feature. Choices=[None, 'log'], default=None. A comma-separeted list of key-value pairs can be used to set the normalization of multiple features. If used, the global normalization method set in genomic_features_norm argument will be overriden for the specified feature/s.

#### - The DNA sequence feature can be excluded.

You can specify that by using the '--no-sequence' flag. 

#### - Training on individual chromosomes.

To facilitate testing you can make use of the '--test_chromosome' argument to train on a specific chromosome. default=None.

#### - Examples of setups in different training strategies:

Standard training:  
--celltype MCG0023 --genomic_features_norm 'log' --feature_norms ctcf:None.

Standard training, no CTCF DNA input:  
--celltype MCG0023 --genomic_features_norm 'log'.

MaxATAC predicted CTCF, multiple-sample:  
--celltype MCG0023,MCG0019,MCG0027,MCG0034 --genomic_features_norm None.

Prior TF activity training with ATAC-seq, no DNA sequence:  
--celltype MCG0023 —-genomic_features_norm 'None' --feature_norms atac:'log' --no_sequence.  
(The TFs activity values are not transformed while the atac-seq is log-transformed).

Prior TF activity training, no ATAC-seq, no DNA sequence:  
--celltype MCG0023  --genomic_features_norm None --no_sequence.


#### Installation:
 
1) Download the pipeline repository.
 
git clone https://github.com/javrodriguez/C.Origami_vGeneric.git

cd C.Origami_vGeneric

2) Create conda environment.

conda create -n corigami_vGeneric python==3.9 pytorch==1.12.0 torchvision==0.13.0 pytorch-cuda=11.8 pandas==1.3.0 matplotlib==3.3.2 pybigwig==0.3.18 omegaconf==2.1.1 tqdm==4.64.0 pytorch-lightning=1.9 scikit-image lightning-bolts mkl==2024.0 -c pytorch -c nvidia

3) Install C.Origami.

conda activate C.Origami_vGeneric
pip install .
