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

We first verified the preprocessing status of the `.h5ad` file before applying any additional transformations.

- All datasets contain the same **36,601 features**
- `adata.X` stores **log-normalized counts per 10,000 counts per cell**
- `adata.layers["UMIs"]` stores **raw counts**

This means:

- **library-size normalization** is already present in `adata.X`
- **log transform** is already present in `adata.X`
- **raw counts** are available in `adata.layers["UMIs"]`

### Highly variable genes

The dataset provides the full feature space of **36,601 genes**. Highly variable genes were **not assumed** to be pre-selected. For models that require HVG selection, we selected the top **2,000 HVGs**.
