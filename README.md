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

```text
=== Donors per split by label ===
label  donors_label0_NotAD  donors_label1_High
split
test                     2                   8
train                    7                  31

=== Cells per split by label ===
label  cells_label0_NotAD  cells_label1_High
split
test                 6495              13736
train               17950              48604

=== train | label=1 (High) ===
Donor ID    n_cells
H20.33.017  6265
H21.33.027  3522
H20.33.008  3389
H20.33.032  3306
H20.33.025  2899
H21.33.017  2510
H20.33.039  2269
H20.33.046  2010
H20.33.031  1975
H21.33.035  1960
...

=== train | label=0 (Not AD) ===
Donor ID    n_cells
H21.33.003  5976
H21.33.004  5134
H20.33.035  2421
H20.33.002  1726
H21.33.023  1414
...

=== test | label=1 (High) ===
Donor ID    n_cells
H21.33.002  2686
H20.33.011  2547
H20.33.030  2406
H20.33.041  1754
...

=== test | label=0 (Not AD) ===
Donor ID    n_cells
H21.33.011  4706
H20.33.044  1789