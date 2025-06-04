### C.Origami vGeneric.

C.Origami modified generic version that attempts to flexibilize the setup of the input features. It can be used to train models with different designs of features.

The input features are not fixed to the standard C.Origami features ATAC-seq, ChIP-CTCF and DNA sequence. In this version the genomic features are not hard-coded, they are dynamically fetched from the /genomic_features/ directory. Each bigwig file found in that directory will be used as a feature. The bigwig file name provides the name of the feature.

The normalization setup of the genomic features can be customized by making use of two arguments; 1) The ‘—genomic_features_norm’ argument sets one same normalization type for all the genomic features. Choices=[None, ‘log'], default=None. 2) The ‘—feature_norms’ argument can set the normalization type of specific features in a dictionary fashion. Key-value pairs where the feature name is the key and the normalization method is the value for that feature. Choices=[None, ‘log'], default=None. 

Flexible setup of genomic samples. The argument ‘—celltype’ now accepts multiple samples by using a comma-separated list of sample names. Example: —celltype MCG0023,MCG0019,MCG0027,MCG0034

Using DNA sequence is not a mandatory. By default it requires DNA sequence. Setting a ‘—no-sequence’ flag excludes this feature. 
