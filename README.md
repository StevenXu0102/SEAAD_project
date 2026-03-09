# SEAAD Oligodendrocyte Classification

## 1) Donor Group Definition

I defined donor groups using the `ADNC` column in `adata.obs`, with `Donor ID` as the donor identifier.

For the binary donor grouping:
- donors were labeled **High** if `ADNC == "High"`
- donors were labeled **Not AD** if `ADNC == "Not AD"`

Donors/cells with other `ADNC` values were excluded from this binary comparison.

## 2) Cell Filtering to Oligodendrocytes

I filtered to Oligodendrocyte cells using `Subclass == "Oligodendrocyte"`, and then assigned donor labels within this filtered subset.

### Donors per group after filtering
- **High:** 39
- **Not AD:** 9

### Cells per group after filtering
- **High:** 62,340
- **Not AD:** 24,445

## 3) Train / Test Split (Donor-Level)

Label = 1: Cells from High donors; Label = 0: Cells from Not AD donors

### Donors per split by label

| Split | Donors (Label 0, Not AD) | Donors (Label 1, High) |
|---|---:|---:|
| Test | 2 | 8 |
| Train | 7 | 31 |

### Cells per split by label

| Split | Cells (Label 0, Not AD) | Cells (Label 1, High) |
|---|---:|---:|
| Test | 6495 | 13736 |
| Train | 17950 | 48604 |

### Train donors with Label = 1 (High)

| Donor ID | n_cells |
|---|---:|
| H20.33.017 | 6265 |
| H21.33.027 | 3522 |
| H20.33.008 | 3389 |
| H20.33.032 | 3306 |
| H20.33.025 | 2899 |
| H21.33.017 | 2510 |
| H20.33.039 | 2269 |
| H20.33.046 | 2010 |
| H20.33.031 | 1975 |
| H21.33.035 | 1960 |
| H20.33.029 | 1795 |
| H21.33.026 | 1637 |
| H21.33.029 | 1555 |
| H21.33.013 | 1483 |
| H20.33.004 | 1354 |
| H20.33.033 | 1343 |
| H20.33.045 | 1186 |
| H20.33.020 | 1147 |
| H21.33.031 | 1031 |
| H20.33.038 | 934 |
| H20.33.028 | 880 |
| H20.33.018 | 878 |
| H20.33.026 | 834 |
| H21.33.040 | 645 |
| H21.33.033 | 591 |
| H20.33.037 | 458 |
| H21.33.046 | 291 |
| H21.33.045 | 182 |
| H20.33.036 | 163 |
| H21.33.020 | 71 |
| H21.33.034 | 41 |

### Train donors with Label = 0 (Not AD)

| Donor ID | n_cells |
|---|---:|
| H21.33.003 | 5976 |
| H21.33.004 | 5134 |
| H20.33.035 | 2421 |
| H20.33.002 | 1726 |
| H21.33.023 | 1414 |
| H21.33.041 | 877 |
| H19.33.004 | 402 |

### Test donors with Label = 1 (High)

| Donor ID | n_cells |
|---|---:|
| H21.33.002 | 2686 |
| H20.33.011 | 2547 |
| H20.33.030 | 2406 |
| H20.33.041 | 1754 |
| H21.33.036 | 1383 |
| H21.33.007 | 1043 |
| H21.33.042 | 961 |
| H21.33.008 | 956 |

### Test donors with Label = 0 (Not AD)

| Donor ID | n_cells |
|---|---:|
| H21.33.011 | 4706 |
| H20.33.044 | 1789 |

## 4) Preprocessing Check

We verified the preprocessing status of the `.h5ad` file before applying any additional transformations.

- All datasets contain the same **36,601 features**
- `adata.X` stores **log-normalized counts per 10,000 counts per cell**
- `adata.layers["UMIs"]` stores **raw counts**

This means:

- **library-size normalization** is already present in `adata.X`
- **log transform** is already present in `adata.X`
- **raw counts** are available in `adata.layers["UMIs"]`

### Highly variable genes

The dataset provides the full feature space of **36,601 genes**. Highly variable genes were **not assumed** to be pre-selected. For models that require HVG selection, we selected the top **2,000 HVGs**.

## 5) Model Training

### A. Baseline MLP

As a simple baseline, I trained a multilayer perceptron (MLP) on per-cell gene expression features. All codes for this baseline are in **`train_mlp.py`**.

#### Architecture

The MLP consists of:

- **2–4 fully connected layers**
- **ReLU** activations after each hidden layer
- optional **dropout**
- optional **BatchNorm**
- a final output layer for binary classification

The output layer can be implemented in either of the following ways:

- **single-logit output** with **sigmoid** activation during evaluation, trained with `BCEWithLogitsLoss`
- **two-class output** with **softmax**, trained with `CrossEntropyLoss`

In my implementation, the MLP takes each cell’s gene expression vector as input and predicts whether the cell comes from a **High** donor (`label = 1`) or a **Not AD** donor (`label = 0`).

#### Input features

The model uses a fixed-length gene expression vector for each cell. Depending on experiments, this input may be:

- the full processed feature vector, or
- a selected subset such as highly variable genes (HVGs)


### B. Transformer-based Model

I also implemented a transformer-based classifier to model cell-level gene expression using sequence-style representations. All codes for this model are in **`train_transformer.py`**.


#### Input representation

For this model, each cell’s gene expression vector is split into chunks and represented as a sequence of embeddings.


#### Cell-level classification representation

The transformer supports **two ways** to construct the final cell-level representation for classification:

1. **CLS token (`pooling="cls"`)**  
   A learned classification token is prepended to the input sequence. After passing through the transformer encoder, the final hidden state of this token is used as the cell-level representation for binary classification. 

2. **Mean pooling (`pooling="mean"`)**  
   Instead of using a dedicated classification token, the model averages the contextualized hidden states of all sequence tokens after the transformer encoder. The resulting pooled embedding is then passed to the classification head.

In our implementation, both options are supported through the `--pooling` argument in **`train_transformer.py`**, with choices:

- `cls`
- `mean`

###  C. Fine-Tuned Foundation Model

I also finetuned scGPT for this task.  All codes for finetuning scGPT are are in **`train_scGPT.py`**.

#### Gene Vocabulary Alignment

For the scGPT model, gene vocabulary alignment is performed before tokenization and training.

#### Steps

1. **Set the dataset gene identifiers**
   - The dataset gene IDs are taken from `adata_sub.var["gene_ids"]`.
   - These gene IDs are then used as the working gene names by setting:
     - `adata_sub.var = adata_sub.var.set_index("gene_ids")`
     - `adata_sub.var["gene_name"] = adata_sub.var.index.astype(str)`

2. **Load the pretrained scGPT vocabulary**
   - The vocabulary is loaded from `vocab.json` using `GeneVocab.from_file(...)`.

3. **Add required special tokens**
   - The following special tokens are ensured to exist in the vocabulary:
     - `<pad>`
     - `<cls>`
     - `<eoc>`

4. **Check which dataset genes are covered by the scGPT vocabulary**
   - For each gene in `adata_sub.var["gene_name"]`, we check whether it exists in the pretrained vocabulary.

5. **Filter to matched genes only**
   - Only genes present in the scGPT vocabulary are kept.


## 6) Evaluation (Test Set)

We evaluated each model on the held-out **donor-level test set** and report **Accuracy** and **F1 score**.

### Test Results

| Model | Best Accuracy | Best F1 |
|---|---:|---:|
| MLP | 0.7245 | 0.8262 |
| Transformer | 0.7952 | 0.8585 |

### Discussion

The task is clearly affected by **class imbalance**. In our dataset, the majority class is **label 1 (High)**, which makes the classification problem biased toward the positive class. Because of this imbalance, accuracy alone is not sufficient, so we also report **F1 score**, which better reflects the balance between precision and recall.

Another important observation is that the models **overfit very quickly**. During training, the training accuracy rises rapidly and becomes close to **100%** after relatively few epochs, while the test performance improves much more slowly and remains substantially lower. This indicates that the models are learning the training donors extremely well but do not generalize equally well to unseen donors.

To address class imbalance, i also implemented **weighted loss functions** for both:

- **BCEWithLogitsLoss**
- **CrossEntropyLoss**

but the improvement was limited. This suggests that the main difficulty is not only label imbalance, but also a combination of factors such as donor-level heterogeneity and strong donor-specific signals.