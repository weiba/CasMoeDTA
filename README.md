# CasMoeDTA

CasMoeDTA is a deep learning model for drug--target affinity (DTA) prediction. The model integrates drug molecular representations, protein sequence representations, and protein functional features, and uses a cascade Mixture-of-Experts framework to model drug--protein interaction patterns.

This repository contains the code for feature extraction, model training, and evaluation on Davis and KIBA datasets.

## Requirements

The main dependencies are:

- python == 3.9
- pytorch == 2.0.1
- torch-geometric == 2.6.1
- numpy == 2.0.2
- pandas == 2.3.3
- scipy == 1.13.1
- scikit-learn == 1.6.1
- transformers == 4.49.0
- rdkit
- gensim
- tqdm

You can install the basic dependencies by:

```bash
pip install -r requirements.txt
```

For GPU training, please install PyTorch and PyTorch Geometric according to your CUDA version. The original environment uses CUDA 11.8.

## Instructions

This project contains all codes for CasMoeDTA, including drug feature extraction, protein feature extraction, protein functional feature construction, model training, and model evaluation.

The main task is DTA regression. The supported datasets are Davis and KIBA. The supported experimental settings are:

- warm_start
- drug_coldstart
- protein_coldstart

## Model composition and meaning

CasMoeDTA is composed of feature extraction modules and model training modules.

### Feature extraction module

- `Mol2Vec/` is used to generate drug molecular representations from SMILES sequences.
  - `Mol2Vec.py` extracts Mol2Vec-based global drug features and atom/substructure-level drug features.
  - `model_300dim.pkl` is the pretrained Mol2Vec model.
  - `chembl_candidate_features/` stores related Mol2Vec feature resources.

- `ProtTransBertBFD_seq_matrix.py` is used to generate protein representations from protein sequences.
  - It extracts protein-level global vectors and residue-level sequence matrices using ProtTransBertBFD.

- `Function/` is used to generate protein functional features.
  - `word_build.py` constructs functional vocabulary or intermediate functional representations.
  - `achieve_function_vector.py` generates protein functional vectors.

### Model and training module

- `LLMDTA4_moetransformer_GRU.py` defines the CasMoeDTA model.
  - It contains the protein functional gated fusion module.
  - It contains the sequence encoder for drug and protein matrices.
  - It contains the MoE-Transformer interaction module.
  - It contains the final MoE-based affinity prediction module.

- `training_moetransformer_noval.py` is the main training and evaluation script.
  - It loads drug, protein, and functional features.
  - It trains CasMoeDTA under different experimental settings.
  - It reports RMSE, MSE, Pearson, Spearman, CI, and rm2.

- `TryAttentionBlock.py` contains several attention-related modules used or reserved in model development.

- `utils.py` contains data loading functions and evaluation metrics.

## Data

The data folder should be organized as follows:

```text
CasMoeDTA/
├── code/
│   ├── Function/
│   ├── Mol2Vec/
│   ├── LLMDTA4_moetransformer_GRU.py
│   ├── ProtTransBertBFD_seq_matrix.py
│   ├── training_moetransformer_noval.py
│   ├── TryAttentionBlock.py
│   └── utils.py
├── dta/
│   ├── davis/
│   └── kiba/
└── environment.yml
```

In this repository, the Davis and KIBA data folders contain compressed files to keep each archive part smaller than 20 MB. For both `davis` and `kiba`, the following folders are compressed as split ZIP archives:

```text
dta/davis/domain.zip      # together with domain.z01, .z02, ... if available
dta/davis/data_folds.zip          # together with data_folds.z01, .z02, ... if available
```

Before running the model, unzip these split archives. Please keep all parts of the same archive in the same directory and extract from the final `.zip` file, not from `.z01`.

After extraction, the expected data structure is:

```text
dta/davis/
├── data_folds/
│   ├── warm_start/
│   ├── drug_coldstart/
│   └── protein_coldstart/
├── domain/
├── ligands_can.csv
├── proteins.csv
└── features/                 

 dta/kiba/
├── data_folds/
│   ├── warm_start/
│   ├── drug_coldstart/
│   └── protein_coldstart/
├── domain/
├── ligands_can.csv
├── proteins.csv
└── features/                
```

If the archive `domain.zip` is extracted as a folder named `domain`, please rename it to `features`, or modify the feature paths in `training_moetransformer_noval.py` and `LLMDTA4_moetransformer_GRU.py` accordingly.

The fold files should be named as:

```text
train_fold_1.csv
test_fold_1.csv
train_fold_2.csv
test_fold_2.csv
...
```

Each fold file contains three columns:

```text
compound_id, protein_id, affinity_label
```

The feature folder should contain the precomputed drug, protein, and functional feature files required by the training script.

## Usage

Run CasMoeDTA with:

```bash
cd code
python training_moetransformer_noval.py dta davis warm_start
```

Other examples:

```bash
python training_moetransformer_noval.py dta davis drug_coldstart
python training_moetransformer_noval.py dta davis protein_coldstart
python training_moetransformer_noval.py dta kiba warm_start
python training_moetransformer_noval.py dta kiba drug_coldstart
python training_moetransformer_noval.py dta kiba protein_coldstart
```

## Training process

The training script performs cross-validation on the selected dataset and setting. For each fold, the script:

1. Loads precomputed drug features, protein features, and protein functional features.
2. Builds the training and test dataloaders.
3. Trains CasMoeDTA with MSE loss.
4. Saves the best model checkpoint.
5. Evaluates the model on the test set.
6. Reports RMSE, MSE, Pearson, Spearman, CI, and rm2.

The result file will be saved in:

```text
../results/
```

## Using your own data

To use your own dataset, prepare the data in the same format as Davis or KIBA:

1. Prepare fold files under `data_folds/`.
2. Generate drug features with the scripts in `Mol2Vec/`.
3. Generate protein sequence features with `ProtTransBertBFD_seq_matrix.py`.
4. Generate protein functional features with the scripts in `Function/`.
5. Modify the feature paths in the training script if necessary.
6. Run `training_moetransformer_noval.py` with the corresponding dataset name and setting.

## Results

The output metrics include:

- RMSE
- MSE
- Pearson
- Spearman
- CI
- rm2

The trained model checkpoints are saved as:

```text
best_<dataset>_<setting>_model_fold<fold>_GRU.pth
```

The cross-validation results are saved as:

```text
../results/<dataset>_<setting>.csv
```

## Contact

If you have any questions about the code or data, please open an issue or contact the authors.
