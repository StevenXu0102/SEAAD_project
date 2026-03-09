# SEAAD Oligodendrocyte Classification:

## 1) Donor Group Definition

I defined donor groups using the `ADNC` column in `adata.obs`, with `Donor ID` as the donor identifier.

For the binary donor grouping:
- donors were labeled **High** if `ADNC == "High"`
- donors were labeled **Not AD** if `ADNC == "Not AD"`

Donors/cells with other `ADNC` values were excluded from this binary comparison.

## 2) Cell Filtering to Oligodendrocytes

I filtered to Oligodendrocyte cells using `Subclass == "Oligodendrocyte"`, and then assigned donor labels within this filtered subset.

## Summary After Filtering

### Donors per group after filtering
- **High:** 39
- **Not AD:** 9

### Cells per group after filtering
- **High:** 62,340
- **Not AD:** 24,445