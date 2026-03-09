#!/usr/bin/env python3
import os
import sys
import json
import time
import copy
import random
import argparse
import atexit
import re
from pathlib import Path
from typing import Dict

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse

# scGPT
import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Tee:
    """
    Write everything to both terminal and file.
    Useful for saving all print() outputs.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_stdout_stderr_logging(run_dir: Path):
    """
    Redirect both stdout and stderr to terminal + log file.
    Use rank-specific log files to avoid collisions under multi-process launch.
    """
    rank = os.environ.get("RANK", "0")
    log_path = run_dir / f"stdout_rank{rank}.log"
    log_f = open(log_path, "a", buffering=1)  # line-buffered

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    sys.stdout = Tee(orig_stdout, log_f)
    sys.stderr = Tee(orig_stderr, log_f)

    atexit.register(log_f.close)
    return log_path


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(data_pt: Dict[str, torch.Tensor], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        SeqDataset(data_pt),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )


def train_one_epoch(
    model,
    loader,
    epoch: int,
    device,
    vocab,
    pad_token: str,
    optimizer,
    criterion,
    scaler,
    use_amp: bool,
    threshold: float = 0.5,
    print_every: int = 200,
):
    model.train()
    total_loss, total_n = 0.0, 0
    tp = fp = fn = tn = 0
    prob_sum = 0.0
    prob_min = 1.0
    prob_max = 0.0
    n_prob = 0

    for step, batch in enumerate(loader, start=1):
        input_gene_ids = batch["gene_ids"].to(device)
        input_values = batch["values"].to(device)
        labels = batch["labels"].to(device).long()

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=True,
                CCE=False,
                MVC=False,
                ECS=False,
            )
            logits = out["cls_output"].view(-1)  # (B,)
            loss = criterion(logits, labels.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        prob1 = torch.sigmoid(logits.detach())
        pred = (prob1 >= threshold).long()

        tp += ((pred == 1) & (labels == 1)).sum().item()
        fp += ((pred == 1) & (labels == 0)).sum().item()
        fn += ((pred == 0) & (labels == 1)).sum().item()
        tn += ((pred == 0) & (labels == 0)).sum().item()

        prob_sum += prob1.sum().item()
        prob_min = min(prob_min, float(prob1.min().item()))
        prob_max = max(prob_max, float(prob1.max().item()))
        n_prob += prob1.numel()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        if step == 1:
            print(f"[train epoch {epoch}] start | batches={len(loader)} | batch_size={bs}")
        if print_every and (step % print_every == 0):
            avg = total_loss / max(total_n, 1)
            print(
                f"[train epoch {epoch}] step {step}/{len(loader)} | "
                f"batch_loss={loss.item():.4f} | avg_loss={avg:.4f}"
            )

    avg_loss = total_loss / max(total_n, 1)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
    pred_ones = (tp + fp) / max(tp + fp + tn + fn, 1)
    prob_mean = prob_sum / max(n_prob, 1)

    print(
        f"[train epoch {epoch}] done | loss={avg_loss:.4f} acc={acc:.4f} f1={f1:.4f} "
        f"| pred_ones={pred_ones:.3f} | prob(mean/min/max)={prob_mean:.3f}/{prob_min:.3f}/{prob_max:.3f} "
        f"| TP={tp} FP={fp} FN={fn} TN={tn}"
    )
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(
    model,
    loader,
    epoch: int,
    device,
    vocab,
    pad_token: str,
    criterion,
    use_amp: bool,
    threshold: float = 0.5,
):
    model.eval()
    total_loss, total_n = 0.0, 0
    tp = fp = fn = tn = 0
    prob_sum = 0.0
    prob_min = 1.0
    prob_max = 0.0
    n_prob = 0

    for _, batch in enumerate(loader, start=1):
        input_gene_ids = batch["gene_ids"].to(device)
        input_values = batch["values"].to(device)
        labels = batch["labels"].to(device).long()

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=True,
                CCE=False,
                MVC=False,
                ECS=False,
            )
            logits = out["cls_output"].view(-1)
            loss = criterion(logits, labels.float())

        prob1 = torch.sigmoid(logits)
        pred = (prob1 >= threshold).long()

        tp += ((pred == 1) & (labels == 1)).sum().item()
        fp += ((pred == 1) & (labels == 0)).sum().item()
        fn += ((pred == 0) & (labels == 1)).sum().item()
        tn += ((pred == 0) & (labels == 0)).sum().item()

        prob_sum += prob1.sum().item()
        prob_min = min(prob_min, float(prob1.min().item()))
        prob_max = max(prob_max, float(prob1.max().item()))
        n_prob += prob1.numel()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    avg_loss = total_loss / max(total_n, 1)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
    pred_ones = (tp + fp) / max(tp + fp + tn + fn, 1)
    prob_mean = prob_sum / max(n_prob, 1)

    print(
        f"[test epoch {epoch}] loss={avg_loss:.4f} acc={acc:.4f} f1={f1:.4f} "
        f"| pred_ones={pred_ones:.3f} | prob(mean/min/max)={prob_mean:.3f}/{prob_min:.3f}/{prob_max:.3f} "
        f"| TP={tp} FP={fp} FN={fn} TN={tn}"
    )
    return avg_loss, acc, f1


def _fmt_float(x: float, nd=6) -> str:
    return format(float(x), f".{nd}g")


def build_run_name(dataset_name: str, args) -> str:
    parts = [
        dataset_name,
        f"{Path(args.load_model).name}",
        f"ep{args.epochs}",
        f"bs{args.batch_size}",
        f"lr{_fmt_float(args.lr)}",
        f"drop{_fmt_float(args.dropout)}",
        f"hvg{args.n_hvg}",
        f"bins{args.n_bins}",
        f"ts{_fmt_float(args.test_size)}",
        f"sch{_fmt_float(args.schedule_ratio)}",
        f"thr{_fmt_float(args.threshold)}",
        f"seed{args.seed}",
        ("amp0" if args.no_amp else "amp1"),
    ]
    name = "_".join(parts)
    name = re.sub(r"[^A-Za-z0-9_\-\.]+", "", name)
    return name[:180]


def main():
    parser = argparse.ArgumentParser("scGPT binary finetune (High vs Not AD, Oligodendrocyte-only)")
    parser.add_argument("--data-path", type=str, required=True, help="Path to .h5ad file")
    parser.add_argument("--load-model", type=str, default="./save/scGPT_human", help="Folder with args.json/vocab.json/best_model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--schedule-ratio", type=float, default=0.9)
    parser.add_argument("--save-eval-interval", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=51)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--max-seq-len", type=int, default=None, help="Default: n_hvg + 1")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true", help="Disable fp16 autocast + GradScaler")
    parser.add_argument("--save-root", type=str, default="./save", help="Where to create run folder")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_name = "SEAAD_Oligodendrocyte_ADNC_binary"
    run_name = build_run_name(dataset_name, args)
    run_dir = Path(args.save_root) / f"dev_{run_name}-{time.strftime('%b%d-%H-%M')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Redirect all print output to terminal + log file
    log_path = setup_stdout_stderr_logging(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("save_dir:", run_dir)
    print("stdout/stderr log:", log_path)

    # scGPT logger file handler
    logger = scg.logger
    scg.utils.add_file_handler(logger, run_dir / "run.log")

    # load data
    print("Reading h5ad files...")
    adata = sc.read(args.data_path)
    print("Finishing reading data. Now preprocessing the data!")

    obs = adata.obs.copy()
    required_cols = ["Subclass", "Donor ID", "ADNC"]
    missing = [c for c in required_cols if c not in obs.columns]
    if missing:
        raise ValueError(f"Missing required obs columns: {missing}")

    obs["Subclass"] = obs["Subclass"].astype(str).str.strip()
    obs["ADNC"] = obs["ADNC"].astype(str).str.strip()
    obs["Donor ID"] = obs["Donor ID"].astype(str)

    # filter
    obs_oligo = obs[obs["Subclass"] == "Oligodendrocyte"].copy()
    obs_oligo_bin = obs_oligo[obs_oligo["ADNC"].isin(["High", "Not AD"])].copy()
    obs_oligo_bin["label"] = obs_oligo_bin["ADNC"].map({"Not AD": 0, "High": 1}).astype(int)

    keep_cells = obs_oligo_bin.index
    adata_sub = adata[keep_cells].copy()
    print(f"Number of total Oligodendrocyte (High/Not AD) cells: {adata_sub.n_obs}")

    adata_sub.obs["ADNC"] = adata_sub.obs["ADNC"].astype(str).str.strip()
    adata_sub.obs["label"] = adata_sub.obs["ADNC"].map({"Not AD": 0, "High": 1}).astype(int)

    n0 = int((adata_sub.obs["label"] == 0).sum())
    n1 = int((adata_sub.obs["label"] == 1).sum())
    print(f"Cells Not AD (0): {n0}")
    print(f"Cells High   (1): {n1}")

    # var
    adata_sub.var = adata_sub.var.set_index("gene_ids")
    adata_sub.var["gene_name"] = adata_sub.var.index.astype(str)

    # pretrained config + vocab
    model_dir = Path(args.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs.get("n_layers_cls", 0)

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    pad_value = -2

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[pad_token])

    adata_sub.var["in_vocab"] = [1 if g in vocab else 0 for g in adata_sub.var["gene_name"]]
    n_in = int(adata_sub.var["in_vocab"].sum())
    print(f"Genes in vocab: {n_in}/{adata_sub.n_vars} (vocab size={len(vocab)})")
    adata_sub = adata_sub[:, adata_sub.var["in_vocab"] == 1].copy()

    # preprocess
    n_hvg = args.n_hvg
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else (n_hvg + 1)

    preprocessor = Preprocessor(
        use_key="UMIs",
        filter_gene_by_counts=3,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=n_hvg,
        hvg_use_key="UMIs",
        hvg_flavor="seurat_v3",
        binning=args.n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(adata_sub, batch_key=None)

    input_layer_key = "X_binned"
    all_counts = (
        adata_sub.layers[input_layer_key].A
        if issparse(adata_sub.layers[input_layer_key])
        else adata_sub.layers[input_layer_key]
    )
    genes = adata_sub.var["gene_name"].tolist()

    # donor split
    donor_table = (
        adata_sub.obs.groupby("Donor ID", observed=True)
        .agg(donor_label=("label", "first"))
        .reset_index()
    )
    donor_ids = donor_table["Donor ID"].to_numpy().astype(str)
    donor_labels = donor_table["donor_label"].to_numpy().astype(int)

    train_donors, test_donors = train_test_split(
        donor_ids,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=donor_labels,
    )
    train_donors = set(train_donors)
    test_donors = set(test_donors)
    print(f"Train donors: {len(train_donors)} | Test donors: {len(test_donors)}")

    donor_per_cell = adata_sub.obs["Donor ID"].to_numpy().astype(str)
    train_mask = np.isin(donor_per_cell, list(train_donors))
    test_mask = np.isin(donor_per_cell, list(test_donors))

    train_data = all_counts[train_mask]
    test_data = all_counts[test_mask]
    all_labels = adata_sub.obs["label"].to_numpy().astype(int)
    train_labels = all_labels[train_mask]
    test_labels = all_labels[test_mask]

    print("train:", train_data.shape, "pos_rate:", float(train_labels.mean()))
    print("test :", test_data.shape, "pos_rate:", float(test_labels.mean()))
    print("train counts:", np.bincount(train_labels))
    print("test  counts:", np.bincount(test_labels))

    # tokenize
    gene_token_ids = np.array(vocab(genes), dtype=int)

    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_token_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    tokenized_test = tokenize_and_pad_batch(
        test_data,
        gene_token_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    print("tokenized train genes:", tokenized_train["genes"].shape)
    print("tokenized test  genes:", tokenized_test["genes"].shape)

    # pack torch tensors
    train_pt = {
        "gene_ids": tokenized_train["genes"],
        "values": tokenized_train["values"],
        "labels": torch.tensor(train_labels, dtype=torch.long),
    }
    test_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": tokenized_test["values"],
        "labels": torch.tensor(test_labels, dtype=torch.long),
    }

    train_loader = prepare_dataloader(train_pt, batch_size=args.batch_size, shuffle=True)
    test_loader = prepare_dataloader(test_pt, batch_size=args.batch_size, shuffle=False)

    # model
    ntokens = len(vocab)
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,  # 1 logit for BCEWithLogitsLoss
        vocab=vocab,
        dropout=args.dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=False,
        n_input_bins=args.n_bins,
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=False,
    )

    if model_file.exists():
        load_pretrained(model, torch.load(model_file, map_location="cpu"), verbose=False)
        print("Loaded pretrained weights:", model_file)
    else:
        print("WARNING: pretrained model_file not found:", model_file)

    for p in model.parameters():
        p.requires_grad_(True)

    model.to(device)

    # loss
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"pos_weight={pos_weight.item():.4f} (neg/pos)")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    use_amp = (not args.no_amp) and (device.type == "cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4 if use_amp else 1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.schedule_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # loop + save best by F1
    best_test_f1 = -1.0
    best_epoch = -1
    best_model = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, epoch, device, vocab, pad_token,
            optimizer, criterion, scaler, use_amp,
            threshold=args.threshold,
            print_every=200,
        )

        test_loss, test_acc, test_f1 = evaluate(
            model, test_loader, epoch, device, vocab, pad_token,
            criterion, use_amp, threshold=args.threshold
        )

        scheduler.step()

        print(
            f"epoch {epoch:3d} | time {time.time()-t0:5.2f}s | "
            f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | train_f1 {train_f1:.4f} | "
            f"test_loss {test_loss:.4f} | test_acc {test_acc:.4f} | test_f1 {test_f1:.4f}"
        )

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        if (epoch % args.save_eval_interval == 0) or (epoch == args.epochs):
            if best_model is not None:
                ckpt_path = run_dir / f"best_model_e{best_epoch}.pt"
                torch.save(best_model.state_dict(), ckpt_path)
                print("Saved:", ckpt_path, "| best_test_f1:", best_test_f1)

    print(f"DONE. best_epoch={best_epoch} best_test_f1={best_test_f1:.4f}")


if __name__ == "__main__":
    main()