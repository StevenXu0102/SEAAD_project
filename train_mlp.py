#!/usr/bin/env python3
import json
import math
import time
import argparse
import logging
from pathlib import Path
from typing import List

import torch
from accelerate import Accelerator

from utils import *
from mymodels import *


def setup_logger(log_file: Path, is_main_process: bool):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # avoid duplicated handlers if rerun in notebook / same process
    if logger.handlers:
        logger.handlers.clear()

    if is_main_process:
        fmt = logging.Formatter("%(asctime)s | %(message)s")

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


def build_run_name(args) -> str:
    # Folder name includes key hyperparameters + split settings
    hd = "-".join(str(x) for x in [int(v) for v in args.hidden_dims.split(",") if v.strip()])
    use_weighted = "true" if args.use_weighted_loss else "false"
    balanced_test = "balanced" if args.balanced_test else "unbalanced"
    name = (
        f"mlp"
        f"_hd{hd}"
        f"_dropout{args.dropout}"
        f"_bn{int(args.use_batchnorm)}"
        f"_loss{args.loss_type}"
        f"_wl{use_weighted}"
        f"_lr{args.lr}"
        f"_wd{args.weight_decay}"
        f"_bs{args.batch_size}"
        f"_ep{args.epochs}"
        f"_seed{args.seed}"
        f"_ts{args.test_size}"
        f"_{balanced_test}"
        f"_tg{args.top_genes}"
        f"_vs{args.val_size_within_train}"
        f"_mp{args.mixed_precision}"
    )
    return name


def main():
    ap = argparse.ArgumentParser("Baseline MLP (Accelerate, bf16)")
    # data
    ap.add_argument("--data-path", required=True, help="Path to .h5ad")
    ap.add_argument("--out-root", required=True, help="Root folder to save runs")

    # split params (prepare_split will be called inside)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--make-val", action="store_true")
    ap.add_argument("--val-size-within-train", type=float, default=0.1)

    # training hyperparams
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dims", type=str, default="1024,256")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use-batchnorm", action="store_true")
    ap.add_argument("--loss-type", choices=["ce", "bce"], default="ce")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    ap.add_argument("--metric-for-best", choices=["f1", "roc_auc", "acc"], default="f1")

    # optional weighted loss
    ap.add_argument("--use-weighted-loss", action="store_true")
    ap.add_argument("--top-genes", type=int, default=2000, help="Number of highly variable genes to select")
    ap.add_argument("--balanced_test", action="store_true")

    args = ap.parse_args()

    run_name = build_run_name(args)
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    logger = setup_logger(out_dir / "training.log", accelerator.is_main_process)
    log = logger.info if accelerator.is_main_process else (lambda *a, **k: None)

    log("Starting run")
    log("Output directory: %s", out_dir)
    log("Arguments:\n%s", json.dumps(vars(args), indent=2))

    if accelerator.is_main_process:
        log("Accelerator device: %s", accelerator.device)
        log("Num processes: %s", accelerator.num_processes)
        log("Mixed precision: %s", accelerator.mixed_precision)

    # Load data
    adata, obs_split, donor_report, cell_report, donor_cell_counts, hvg_genes = prepare_split(
        data_path=args.data_path,
        out_csv=str(out_dir / "oligo_cells_with_split.csv"),
        out_meta_json=str(out_dir / "split_meta.json"),
        random_seed=args.random_seed,
        test_size=args.test_size,
        make_val=args.make_val,
        val_size_within_train=args.val_size_within_train,
        n_top_genes=args.top_genes,
        balanced_test=args.balanced_test,
    )

    log("\n=== Donor report ===")
    log("%s", donor_report)
    log("\n=== Cell report ===")
    log("%s", cell_report)
    log("\n=== Cells per donor ===")
    log("%s", donor_cell_counts.to_string(index=False))

    adata = adata[:, hvg_genes]

    # Build datasets
    train_ds, val_ds, test_ds, input_dim = generate_dataset(adata, obs_split)

    log("Dataset summary:")
    log("  input_dim=%d", input_dim)
    log("  train_n=%d", len(train_ds))
    log("  val_n=%s", len(val_ds) if val_ds is not None else "None")
    log("  test_n=%d", len(test_ds))

    # DataLoaders
    train_loader = generate_loader(train_ds, args.batch_size, True, args.num_workers, args.seed)
    val_loader = generate_loader(val_ds, args.batch_size, False, args.num_workers, args.seed) if val_ds is not None else None
    test_loader = generate_loader(test_ds, args.batch_size, False, args.num_workers, args.seed)

    def parse_hidden_dims(s: str) -> List[int]:
        vals = [int(x.strip()) for x in s.split(",") if x.strip()]
        return vals

    # Model / loss / optimizer
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = MLPBaseline(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        use_bce=(args.loss_type == "bce"),
        dropout=args.dropout,
        use_batchnorm=args.use_batchnorm,
    )

    loss_fn = get_loss_fn(args.loss_type, use_weighted=args.use_weighted_loss, y_train=train_ds.y)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Accelerate prepare
    if val_loader is None:
        model, optimizer, train_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, test_loader
        )
    else:
        model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader
        )

    if args.loss_type == "ce" and getattr(loss_fn, "weight", None) is not None:
        loss_fn.weight = loss_fn.weight.to(accelerator.device)
    if args.loss_type == "bce" and getattr(loss_fn, "pos_weight", None) is not None:
        loss_fn.pos_weight = loss_fn.pos_weight.to(accelerator.device)

    history = []
    best_metric = -float("inf")
    best_epoch = -1
    best_ckpt_path = out_dir / "best_model.pt"

    def fmt(m):
        if m is None:
            return "None"
        return (
            f"loss={m['loss']:.4f}, "
            f"acc={m['acc']:.4f}, "
            f"f1={m['f1']:.4f}, "
            f"roc_auc={m['roc_auc']:.4f}, "
            f"pred_ones={m['preds_one']:.2f}, "
            f"n={m['n']}"
        )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, accelerator, args.loss_type
        )

        if val_loader is not None:
            val_metrics = evaluate(
                model, val_loader, loss_fn, accelerator, args.loss_type
            )
        else:
            val_metrics = None

        test_metrics = evaluate(
            model, test_loader, loss_fn, accelerator, args.loss_type
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "time_sec": round(time.time() - t0, 2),
        }
        history.append(record)

        # choose best using val if available, otherwise test
        ref = val_metrics if val_metrics is not None else test_metrics
        current_metric = ref.get(args.metric_for_best, float("nan"))
        if not (isinstance(current_metric, float) and math.isnan(current_metric)) and current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch

            unwrapped = accelerator.unwrap_model(model)
            accelerator.save(
                {
                    "model_state_dict": unwrapped.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "metric_name": args.metric_for_best,
                    "input_dim": input_dim,
                    "hidden_dims": hidden_dims,
                },
                best_ckpt_path,
            )
            log(
                "New best model saved | epoch=%d | %s=%.4f | path=%s",
                best_epoch,
                args.metric_for_best,
                best_metric,
                best_ckpt_path,
            )

        log("\nEpoch %d/%d  time=%.2fs", epoch, args.epochs, record["time_sec"])
        log("  train: %s", fmt(train_metrics))
        if val_metrics is not None:
            log("  val  : %s", fmt(val_metrics))
        log("  test : %s", fmt(test_metrics))
        log("  best_%s=%.4f @ epoch %d", args.metric_for_best, best_metric, best_epoch)

        accelerator.wait_for_everyone()

    def save_json(obj, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    if accelerator.is_main_process:
        save_json(history, out_dir / "history.json")
        summary = {
            "best_epoch": best_epoch,
            "best_metric_name": args.metric_for_best,
            "best_metric_value": best_metric,
            "final_epoch": args.epochs,
            "final_train": history[-1]["train"],
            "final_val": history[-1]["val"],
            "final_test": history[-1]["test"],
            "best_model_path": str(best_ckpt_path),
            "log_path": str(out_dir / "training.log"),
        }
        save_json(summary, out_dir / "summary.json")

        log("\n=== Final Test Report ===")
        log("%s", json.dumps(history[-1]["test"], indent=2))
        log("Saved best model: %s", best_ckpt_path)
        log("Saved history: %s", out_dir / "history.json")
        log("Saved summary: %s", out_dir / "summary.json")
        log("Saved log: %s", out_dir / "training.log")


if __name__ == "__main__":
    main()