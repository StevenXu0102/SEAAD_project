#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from RNA_datasets import RNADataset
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from accelerate import Accelerator

def _assert_no_donor_overlap(obs_split: pd.DataFrame):
    donor_split_nunique = obs_split.groupby("Donor ID")["split"].nunique()
    bad = donor_split_nunique[donor_split_nunique > 1]
    if len(bad) > 0:
        raise AssertionError(
            f"Donor overlap across multiple splits detected: {bad.head(10).to_dict()}"
        )

    train_donors = set(obs_split.loc[obs_split["split"] == "train", "Donor ID"].astype(str).unique())
    test_donors = set(obs_split.loc[obs_split["split"] == "test", "Donor ID"].astype(str).unique())
    overlap = train_donors & test_donors
    if len(overlap) > 0:
        raise AssertionError(f"Train/test donor overlap detected: {list(sorted(overlap))[:10]}")

def _make_reports(obs_split: pd.DataFrame):
    donor_report = (
        obs_split.groupby(["split", "label"])["Donor ID"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns={0: "donors_label0_NotAD", 1: "donors_label1_High"})
        .sort_index()
    )

    cell_report = (
        obs_split.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "cells_label0_NotAD", 1: "cells_label1_High"})
        .sort_index()
    )

    donor_cell_counts = (
        obs_split.groupby(["split", "Donor ID"])
        .size()
        .reset_index(name="n_cells")
        .sort_values(["split", "n_cells", "Donor ID"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    return donor_report, cell_report, donor_cell_counts

def _balanced_donor_split(
    donor_ids: List[str],
    donor_labels: List[int],          # 0=NotAD, 1=High
    test_size: float,
    random_seed: int,
    test_donors_per_class: Optional[int] = None,
    make_val: bool = False,
    val_size_within_train: float = 0.1,
    balanced_val: bool = True,        # if True, val has equal donors per class
) -> Tuple[List[str], List[str], Optional[List[str]], int]:
    
    if len(donor_ids) != len(donor_labels):
        raise ValueError("donor_ids and donor_labels must have the same length")

    # split donors by label
    notad = [str(d) for d, y in zip(donor_ids, donor_labels) if int(y) == 0]
    high  = [str(d) for d, y in zip(donor_ids, donor_labels) if int(y) == 1]
    if len(notad) == 0 or len(high) == 0:
        raise ValueError(f"Need both classes present. Got NotAD={len(notad)}, High={len(high)}")

    rng = random.Random(random_seed)
    rng.shuffle(notad)
    rng.shuffle(high)

    # choose k for balanced test
    if test_donors_per_class is None:
        # target total test donors ≈ test_size * total_donors, but must be 2k
        target_total = int(round(test_size * (len(notad) + len(high))))
        target_total = max(2, target_total)
        k = max(1, target_total // 2)
    else:
        k = int(test_donors_per_class)

    # ensure test has donors from both classes AND leaves at least 1 donor of each class for training
    k = min(k, len(notad) - 1, len(high) - 1)
    if k < 1:
        raise ValueError(
            f"Cannot form balanced test while leaving at least 1 donor per class in train. "
            f"Counts: NotAD={len(notad)}, High={len(high)}"
        )

    test_donors = notad[:k] + high[:k]
    train_donors = notad[k:] + high[k:]

    val_donors = None
    if make_val:
        # make a donor-level val split from the train donors
        train_notad = notad[k:]
        train_high  = high[k:]

        if balanced_val:
            rng.shuffle(train_notad)
            rng.shuffle(train_high)

            target_val_total = int(round(val_size_within_train * len(train_donors)))
            target_val_total = max(2, target_val_total)
            kv = max(1, target_val_total // 2)
            kv = min(kv, len(train_notad) - 1, len(train_high) - 1)  # leave at least 1/class in train
            if kv < 1:
                raise ValueError(
                    f"Cannot form balanced val while leaving at least 1 donor per class in train. "
                    f"After test split: train_NotAD={len(train_notad)}, train_High={len(train_high)}"
                )

            val_donors = train_notad[:kv] + train_high[:kv]
            val_set = set(val_donors)
            train_donors = [d for d in train_donors if d not in val_set]
        else:
            # stratified val split (keeps imbalance)
            tr_ids = train_donors
            tr_labels = [0 if d in set(train_notad) else 1 for d in tr_ids]
            train2, val = train_test_split(
                tr_ids,
                test_size=val_size_within_train,
                random_state=random_seed,
                stratify=tr_labels,
            )
            train_donors, val_donors = list(map(str, train2)), list(map(str, val))

    # sanity: no overlaps
    tr_set, te_set = set(train_donors), set(test_donors)
    if tr_set & te_set:
        raise RuntimeError("Train/test donor overlap detected.")
    if make_val and val_donors is not None:
        va_set = set(val_donors)
        if (tr_set & va_set) or (te_set & va_set):
            raise RuntimeError("Train/val/test donor overlap detected.")

    return train_donors, test_donors, val_donors, k

def prepare_split(
    data_path: str,
    out_csv: Optional[str] = None,
    out_meta_json: Optional[str] = None,
    random_seed: int = 42,
    test_size: float = 0.2,
    make_val: bool = False,
    val_size_within_train: float = 0.1,
    n_top_genes: int = 2000,
    balanced_test: bool = True,
    test_donors_per_class: Optional[int] = None,
    balanced_val: bool = True,
):
    """
    Fixed rules (hardcoded):
      - keep only Subclass == 'Oligodendrocyte'
      - keep only ADNC in {'High', 'Not AD'}
      - donor column = 'Donor ID'
      - label map: Not AD -> 0, High -> 1
    """

    print("Loading data...")
    # Load .h5ad (metadata only)
    adata = sc.read_h5ad(data_path, backed="r")
    obs = adata.obs.copy()
    obs["obs_name"] = obs.index.astype(str)

    required_cols = ["Subclass", "Donor ID", "ADNC"]
    missing = [c for c in required_cols if c not in obs.columns]
    if missing:
        raise ValueError(f"Missing required obs columns: {missing}")

    # Filter to Oligodendrocyte FIRST
    obs["Subclass"] = obs["Subclass"].astype(str).str.strip()
    obs_oligo = obs[obs["Subclass"] == "Oligodendrocyte"].copy()

    # Keep only High vs Not AD
    obs_oligo["ADNC"] = obs_oligo["ADNC"].astype(str).str.strip()
    obs_oligo_bin = obs_oligo[obs_oligo["ADNC"].isin(["High", "Not AD"])].copy()
    obs_oligo_bin["label"] = obs_oligo_bin["ADNC"].map({"Not AD": 0, "High": 1}).astype(int)

    # Donor table
    donor_table = (
        obs_oligo_bin.groupby("Donor ID", observed=True)
        .agg(donor_label=("label", "first"))
        .reset_index()
    )
    donor_table["Donor ID"] = donor_table["Donor ID"].astype(str)
    donor_table["donor_label"] = donor_table["donor_label"].astype(int)

    # Train/test split
    donor_ids = donor_table["Donor ID"].to_numpy()
    donor_labels = donor_table["donor_label"].to_numpy()

    print("Spliting data...")
    if balanced_test:
        train_donors, test_donors, val_donors, k_test = _balanced_donor_split(
            donor_ids=list(donor_ids),
            donor_labels=list(donor_labels),
            test_size=test_size,
            random_seed=random_seed,
            test_donors_per_class=test_donors_per_class,
            make_val=make_val,
            val_size_within_train=val_size_within_train,
            balanced_val=balanced_val,
        )
        split_map: Dict[str, str] = {str(d): "train" for d in train_donors}
        if make_val and val_donors is not None:
            split_map.update({str(d): "val" for d in val_donors})
        split_map.update({str(d): "test" for d in test_donors})
    else:
        train_donors, test_donors, _, _ = train_test_split(
            donor_ids,
            donor_labels,
            test_size=test_size,
            random_state=random_seed,
            stratify=donor_labels,
        )

        split_map: Dict[str, str] = {str(d): "train" for d in train_donors}
        split_map.update({str(d): "test" for d in test_donors})

        if make_val:
            train_donor_df = donor_table[donor_table["Donor ID"].isin(train_donors)].copy()
            tr_ids = train_donor_df["Donor ID"].to_numpy()
            tr_labels = train_donor_df["donor_label"].to_numpy()

            train2_donors, val_donors, _, _ = train_test_split(
                tr_ids,
                tr_labels,
                test_size=val_size_within_train,
                random_state=random_seed,
                stratify=tr_labels,
            )

            split_map = {str(d): "train" for d in train2_donors}
            split_map.update({str(d): "val" for d in val_donors})
            split_map.update({str(d): "test" for d in test_donors})

    obs_split = obs_oligo_bin[["obs_name", "Donor ID", "ADNC", "Subclass", "label"]].copy()
    obs_split["Donor ID"] = obs_split["Donor ID"].astype(str)
    obs_split["split"] = obs_split["Donor ID"].map(split_map)
    obs_split = obs_split.dropna(subset=["split"]).copy()

    # Sanity checks
    _assert_no_donor_overlap(obs_split)
    donor_report, cell_report, donor_cell_counts = _make_reports(obs_split)

    for sp in ["train", "test"]:
        if sp in donor_report.index:
            row = donor_report.loc[sp]
            if row.get("donors_label0_NotAD", 0) == 0 or row.get("donors_label1_High", 0) == 0:
                raise AssertionError(f"Split '{sp}' does not contain both labels:\n{row}")

    print("Selecting highly variable genes (HVGs)...")
    # HVG selection using train cells only
    train_obs_names = (
        obs_split.loc[obs_split["split"] == "train", "obs_name"]
        .astype(str)
        .to_numpy()
    )
    if len(train_obs_names) == 0:
        raise ValueError("No train cells found; cannot compute HVGs.")

    ad_train = adata[train_obs_names, :].to_memory()
    sc.pp.highly_variable_genes(
        ad_train,
        n_top_genes=n_top_genes,
        subset=False,
        inplace=True,
    )

    hv = ad_train.var["highly_variable"].to_numpy().astype(bool)
    hvg_genes = ad_train.var_names[hv].astype(str).tolist()

    # Save outputs
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        obs_split.to_csv(out_csv, index=False)

    if out_meta_json is not None:
        Path(out_meta_json).parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "data_path": data_path,
            "fixed_rules": {
                "cell_filter": {"column": "Subclass", "value": "Oligodendrocyte"},
                "diagnosis_filter": {"column": "ADNC", "keep": ["High", "Not AD"]},
                "donor_column": "Donor ID",
                "label_map": {"Not AD": 0, "High": 1},
            },
            "split_rule": {
                "unit": "donor",
                "random_seed": random_seed,
                "test_size": test_size,
                "make_val": make_val,
                "val_size_within_train": val_size_within_train if make_val else None,
                "balanced_test": balanced_test,
                "test_donors_per_class": test_donors_per_class,
                "balanced_val": balanced_val if make_val else None,
            },
            "n_cells_after_filtering": int(len(obs_split)),
            "n_donors_after_filtering": int(obs_split["Donor ID"].nunique()),
            "donor_report": donor_report.to_dict(),
            "cell_report": cell_report.to_dict(),
        }
        with open(out_meta_json, "w") as f:
            json.dump(meta, f, indent=2)

    try:
        adata.file.close()
    except Exception:
        pass

    return adata, obs_split, donor_report, cell_report, donor_cell_counts, hvg_genes

def generate_dataset(adata, obs_split: pd.DataFrame):
    print("Preparing datasets...")
    df = obs_split.copy()
    df["obs_name"] = df["obs_name"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    adata_index = pd.Index(adata.obs_names.astype(str))
    row_idx = adata_index.get_indexer(df["obs_name"].to_numpy())
    if (row_idx < 0).any():
        bad = df.loc[row_idx < 0, "obs_name"].head(10).tolist()
        raise ValueError(f"Some obs_name not found in adata.obs_names. Examples: {bad}")

    X_all = adata.X[row_idx, :]  # assume sparse
    y_all = df["label"].to_numpy(dtype=np.int64)
    s_all = df["split"].to_numpy()

    train_mask = s_all == "train"
    val_mask = s_all == "val"
    test_mask = s_all == "test"

    train_ds = RNADataset(X_all[train_mask], y_all[train_mask])
    val_ds = RNADataset(X_all[val_mask], y_all[val_mask]) if val_mask.any() else None
    test_ds = RNADataset(X_all[test_mask], y_all[test_mask])
     
    # Print dataset lengths
    print("Finish generating dataset...")
    print(f"[generate_dataset] Total matched samples: {len(df)}")
    print(f"[generate_dataset] Train size: {len(train_ds)}")
    print(f"[generate_dataset] Val size: {len(val_ds) if val_ds is not None else 0}")
    print(f"[generate_dataset] Test size: {len(test_ds)}")
    print(f"[generate_dataset] #Features (genes): {X_all.shape[1]}")
    
    return train_ds, val_ds, test_ds, X_all.shape[1]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def generate_loader(dataset, batch_size, shuffle, num_workers, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

def train_one_epoch(model, loader, optimizer, loss_fn, accelerator: Accelerator, loss_type: str, gene_mask_p: float = 0.3):
    model.train()
    total_loss = 0.0
    total_n = 0

    all_y = []
    all_prob = []
    all_pred = []

    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)

        # masking
        if gene_mask_p > 0:
            mask = (torch.rand_like(x) > gene_mask_p).to(x.dtype)
            x = x * mask

        with accelerator.autocast():
            logits = model(x)

            if loss_type == "ce":
                loss = loss_fn(logits, y)
                prob1 = torch.softmax(logits, dim=1)[:, 1]
                pred = torch.argmax(logits, dim=1)
            else:
                # BCE: output shape [B,1] or [B]
                logits = logits.view(-1)
                y_float = y.float()
                loss = loss_fn(logits, y_float)
                prob1 = torch.sigmoid(logits)
                pred = (prob1 >= 0.5).long()

        accelerator.backward(loss)
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.detach().item() * bs
        total_n += bs

        # gather metrics
        y_g, prob_g, pred_g = accelerator.gather_for_metrics((y.detach(), prob1.detach(), pred.detach()))
        all_y.append(y_g.cpu().numpy())
        all_prob.append(prob_g.cpu().numpy())
        all_pred.append(pred_g.cpu().numpy())

    loss_sum_t = torch.tensor(total_loss, device=accelerator.device)
    n_sum_t = torch.tensor(total_n, device=accelerator.device)
    loss_sum_t = accelerator.reduce(loss_sum_t, reduction="sum")
    n_sum_t = accelerator.reduce(n_sum_t, reduction="sum")
    mean_loss = (loss_sum_t / n_sum_t.clamp(min=1)).item()

    y_true = np.concatenate(all_y) if all_y else np.array([], dtype=np.int64)
    y_prob = np.concatenate(all_prob) if all_prob else np.array([], dtype=np.float32)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)

    preds_one = float(np.mean(y_pred == 1)) if len(y_pred) else float("nan")
    roc = float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_prob))

    metrics = { 
        "loss": float(mean_loss),
        "acc": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else float("nan"),
        "roc_auc": roc,
        "preds_one": preds_one,
        "n": int(len(y_true)),
    }
    return metrics

@torch.no_grad()
def evaluate(model, loader, loss_fn, accelerator: Accelerator, loss_type: str):
    model.eval()
    total_loss = 0.0
    total_n = 0

    all_y = []
    all_prob = []
    all_pred = []

    for x, y in loader:
        with accelerator.autocast():
            logits = model(x)

            if loss_type == "ce":
                loss = loss_fn(logits, y)
                prob1 = torch.softmax(logits, dim=1)[:, 1]
                pred = torch.argmax(logits, dim=1)
            else:
                logits = logits.view(-1)
                y_float = y.float()
                loss = loss_fn(logits, y_float)
                prob1 = torch.sigmoid(logits)
                pred = (prob1 >= 0.5).long()

        bs = y.size(0)
        total_loss += loss.detach().item() * bs
        total_n += bs

        y_g, prob_g, pred_g = accelerator.gather_for_metrics((y.detach(), prob1.detach(), pred.detach()))
        all_y.append(y_g.cpu().numpy())
        all_prob.append(prob_g.cpu().numpy())
        all_pred.append(pred_g.cpu().numpy())

    loss_sum_t = torch.tensor(total_loss, device=accelerator.device)
    n_sum_t = torch.tensor(total_n, device=accelerator.device)
    loss_sum_t = accelerator.reduce(loss_sum_t, reduction="sum")
    n_sum_t = accelerator.reduce(n_sum_t, reduction="sum")
    mean_loss = (loss_sum_t / n_sum_t.clamp(min=1)).item()

    y_true = np.concatenate(all_y) if all_y else np.array([], dtype=np.int64)
    y_prob = np.concatenate(all_prob) if all_prob else np.array([], dtype=np.float32)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    
    preds_one = float(np.mean(y_pred == 1)) if len(y_pred) else float("nan")
    roc = float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_prob))

    metrics = {
        "loss": float(mean_loss),
        "acc": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else float("nan"),
        "roc_auc": roc,
        "preds_one": preds_one,
        "n": int(len(y_true)),
    }
    return metrics


def get_loss_fn(loss_type: str, use_weighted: bool = False, y_train: Optional[np.ndarray] = None):
    if (not use_weighted) or (y_train is None):
        return nn.CrossEntropyLoss() if loss_type == "ce" else nn.BCEWithLogitsLoss()

    y_train = np.asarray(y_train, dtype=np.int64)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return nn.CrossEntropyLoss() if loss_type == "ce" else nn.BCEWithLogitsLoss()

    if loss_type == "ce":
        # w_c = N / (2 * n_c)
        N = n_pos + n_neg
        w_neg = N / (2.0 * n_neg)
        w_pos = N / (2.0 * n_pos)
        class_w = torch.tensor([w_neg, w_pos], dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=class_w)

    elif loss_type == "bce":
        # pos_weight = n_neg / n_pos
        pos_weight = torch.tensor([n_neg / float(n_pos)], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
