# fiPIP (Functionally Informed PIPs)
This repository aims to accomplish two tasks for users with statistical fine-mapping results: **(1)** provide a starting point for users seeking to generate or access deep-leaning based sequence-to-omics scores from AlphaGenome, Borzoi, Enformer and/or Sei, and **(2)** generate functionally-informed posterior inclusion probabilities (fiPIPs) from quantitative scores containing functional information. These two tasks can be completed independently of each other. Users can use *any* quantitative scores to generate fiPIPs using this code respository, including those from tools not mentioned in this code repository. In fact, as sequence-to-omics models update and new ones are released, we encourage users to do so. This repository may not be updated if updates are released for the aforementioned sequence-to-omics models or as new ones are released.

## Installation

```bash
python -m pip install -U pip setuptools wheel
pip install -e .
```

## Task 1: Generate or access deep-learning based sequence-to-omics scores

**Please refer to each method's individual installation instructions before use. To prevent conflicts across the different methods which have different requirements, we recommend making virtual or Conda environments for each method.**

### [AlphaGenome](https://github.com/google-deepmind/alphagenome)

Currently, an API has been released for AlphaGenome access. An API key is required. As this is a new method, we recommend following the most up-to-date tutorial; however, we do provide an example script for generating AlphaGenome RNA-seq scores in the tutorials folder, which is a condensed version of their tutorial [here](https://colab.research.google.com/github/google-deepmind/alphagenome/blob/main/colabs/quick_start.ipynb). Please make sure to set your API key before use.

```bash
# Example
pip install alphagenome
export ALPHAGENOME_API_KEY='YOUR_ALPHAGENOME_API_KEY'
fipip alphagenome --input tutorials/example_data.tsv --output alphagenome_results.csv --sep "\t"
```

The output file will have predictions for 667 RNA-seq tracks per variant. The "fallback" column is 1 for a variant if the variant's associated Ensembl ID was not present in the AlphaGenome output, and consequently a mean was taken over all other variants, and 0 otherwise.

### [Borzoi](https://github.com/calico/borzoi)

#### [Pre-computed Borzoi scores](https://console.cloud.google.com/storage/browser/seqnn-share/sniff;tab=objects?prefix=&forceOnObjectsSortingFiltering=false) (recommended)

**Please note that the pre-computed Borzoi scores are based on the hg19 genome build. If your variants are based on the hg38 genome build, please liftover first before continuing.**

With the release of [Srivastava, D. et al. (2025)](https://www.biorxiv.org/content/10.1101/2025.07.09.663936v1.full-text), pre-computed Borzoi scores have been released for over 19 million common and low frequency varaints. While offering less flexibility than generating your own Borzoi scores, using these scores can be efficient and cost effective. Scores are available for both variant effect predictions (VEPs) and principal components (PCs) derived from VEPs.

#### Generating your own Borzoi scores

In order to generate your own Borzoi scores, please follow the installation instructions in the Borzoi repository to download the Borzoi models for use.

Two scripts in the tutorials folder can be used for generating Borzoi scores for the variants in your credible set. The first example script we provide predicts for all 89 RNA-seq tracks. This first script produces two [pickle](https://docs.python.org/3/library/pickle.html) objects per variant, one for each allele, corresponding to RNA-seq predictions at 32 base pair resolution for all 89 tissues across 4 folds.

```bash
# Example
fipip borzoi_1 --input tutorials/example_variants.tsv --outdir borzoi_objects
```

The second script takes the output folder of pickle objects and converts each pickle object to a singular Borzoi score for each variant for each track. If you would like to make predictions for only a subset of tracks, perhaps one(s) more relevant to the tissue of your eQTLs, the 89 columns of the pickle object correspond to the GTEx tissue replicates listed [here](https://github.com/calico/borzoi/blob/5c9358222b5026abb733ed5fb84f3f6c77239b37/examples/targets_gtex.txt). Please set the `--tracks` parameter to make predictions for only a subset of tissues. To make gene-contextual predictions for variants, please provide a GTF file and a file detailing the gene associated with each variant. Otherwise, gene-agnostic predictions will be made. This can be done by setting `--no-gtf`.

```bash
# Example
fipip borzoi_2 --input borzoi_objects --output borzoi_scores.csv --tracks 1-89 --gtf-path /path/to/your/gtf.gtf --gene-map tutorials/example_data.tsv
```

### [Enformer](https://github.com/google-deepmind/deepmind-research/tree/master/enformer)

**Please note that the pre-computed Enformer scores are based on the hg19 genome build. If your variants are based on the hg38 genome build, please liftover first before continuing.**

Pre-computed Enformer scores are available [here](https://console.cloud.google.com/storage/browser/dm-enformer/variant-scores/1000-genomes/enformer). We provide a script for extracting Enformer scores from the h5 files as a script in the tutorials folder.

We currently do not provide a script for generating your own Enformer scores; however, instructions for doing so and example Google Colab notebooks are available in the [Enformer github repository](https://github.com/google-deepmind/deepmind-research/tree/master/enformer).

```bash
# Example
fipip enformer --output enformer_master.csv --h5-dir /path/to/downloaded/h5/files --variants-file tutorials/example_variants.tsv --targets-file tutorials/enformer_targets.txt
```

### [Sei](https://github.com/FunctionLab/sei-framework)

We recommend following the setup instructions and using the chromatin profile prediction and sequence class prediction scripts `1_variant_effect_prediction.sh` and `2_varianteffect_sc_score.sh` respectively in the Sei repository to obtain epigenomic readout and sequence class Sei scores. Both epigenomic readout and sequence class scores are quantitative scores that can be used for fiPIP generation.

## Task 2: Generate functionally informed PIPs (fiPIPs)

We provide a command-line tool for generating fiPIPs from quantitative scores. Please refer to task 1 for direction on how quantitative scores can be obtained if necessary.

Please provide a file for testing and a file for training according to the following format.

### Training file format
Required columns (include column names, use following column names for first three columns):
* variant — Variant ID
* label — Binary 0/1 for negative/positive label respectively
* chr — Variant's chromosome
* Continous scores (any number of columns; column names can be whatever, please make sure they match the names in the testing file)

### Testing file format
Required columns (include column names, use following column names for first four columns):
* variant — Variant ID
* cs_id — Credible set ID
* pip — Posterior inclusion probability (PIP) from statistical fine-mapping
* Continous scores (Please make sure they match the columns in the training file)

The following command will generate PIP-agnostic probability-scale predictions, fiPIPs, and JSON files after training a XGBoost model for each chromosome to the output directory set by `--outdir`:
```bash
fipip calculate_fipip \
  --train-file tutorials/train_df.tsv \
  --predict-file tutorials/test_df.tsv \
  --outdir output \
  --groups 1-5   # optional parameter; 1-based indices; allows subset of continuous scores to be used
```

The following command will generate PIP-agnostic probability-scale predictions and fiPIPs from previously generated XGBoost model (JSON files) to the output directory set by `--outdir`:
```bash
fipip predict_from_json \
  --predict-file tutorials/test_df2.tsv \
  --models-dir output \
  --outdir new_output \
```

