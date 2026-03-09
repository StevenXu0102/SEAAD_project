import json
import math
import time
import argparse
from pathlib import Path
import torch
from accelerate import Accelerator
from utils import *
from mymodels import *


def build_run_name(args) -> str:
    if args.make_val:
        val_size = args.val_size_within_train
    else:
        val_size = 0
    use_weighted = "true" if args.use_weighted_loss else "false"
    balanced_test = "balanced" if args.balanced_test else "unbalanced"
    name = (
        f"transformer"
        f"_dm{args.d_model}"
        f"_nh{args.nhead}"
        f"_nl{args.num_layers}"
        f"_ff{args.dim_feedforward}"
        f"_ck{args.chunk_size}"
        f"_pool{args.pooling}"
        f"_dropout{args.dropout}"
        f"_loss{args.loss_type}"
        f"_wl{use_weighted}"
        f"_lr{args.lr}"
        f"_wd{args.weight_decay}"
        f"_bs{args.batch_size}"
        f"_ep{args.epochs}"
        f"_seed{args.seed}"
        f"_ts{args.test_size}"
        f"_{balanced_test}"
        f"_vs{val_size}"
        f"_tg{args.top_genes}"
        f"_mp{args.mixed_precision}"
    )
    return name

def main():
    ap = argparse.ArgumentParser("Transformer (Accelerate, bf16)")

    # data
    ap.add_argument("--data-path", required=True, help="Path to .h5ad")
    ap.add_argument("--out-root", required=True, help="Root folder to save runs")

    # split params
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--make-val", action="store_true")
    ap.add_argument("--val-size-within-train", type=float, default=0.1)

    # training
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--loss-type", choices=["ce", "bce"], default="ce")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    ap.add_argument("--metric-for-best", choices=["f1", "roc_auc", "acc"], default="f1")

    # transformer params
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dim-feedforward", type=int, default=1024)
    ap.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    ap.add_argument("--norm-first", action="store_true")
    ap.add_argument("--pooling", choices=["cls", "mean"], default="cls")

    ap.add_argument("--use-weighted-loss", action="store_true")
    ap.add_argument("--top-genes", type=int, default=2000, help="Number of highly variable genes to select")
    ap.add_argument("--balanced_test", action="store_true")
    
    args = ap.parse_args()

    run_name = build_run_name(args)
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if accelerator.is_main_process:
        print("Accelerator device:", accelerator.device)
        print("Num processes:", accelerator.num_processes)
        print("Mixed precision:", accelerator.mixed_precision)

    # Enable SDPA backends (flash when possible)
    if torch.cuda.is_available():
        try:
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
            if accelerator.is_main_process and hasattr(torch.backends.cuda, "flash_sdp_enabled"):
                print("flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
        except Exception:
            pass

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

    print("\n=== Donor report ===")
    print(donor_report)
    print("\n=== Cell report ===")
    print(cell_report)

    adata = adata[:, hvg_genes]
    train_ds, val_ds, test_ds, input_dim = generate_dataset(adata, obs_split)

    train_loader = generate_loader(train_ds, args.batch_size, True, args.num_workers, args.seed)
    val_loader = generate_loader(val_ds, args.batch_size, False, args.num_workers, args.seed) if val_ds is not None else None
    test_loader = generate_loader(test_ds, args.batch_size, False, args.num_workers, args.seed)

    # Model (Transformer)
    model = RNATransformer(
        input_dim=input_dim,
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_classes=2,
        use_bce=(args.loss_type == "bce"),
        activation=args.activation,
        norm_first=args.norm_first,
        pooling=args.pooling,
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
    
    if args.loss_type == 'ce' and loss_fn.weight is not None:
        loss_fn.weight = loss_fn.weight.to(accelerator.device)
    if args.loss_type == 'bce' and loss_fn.pos_weight is not None:
        loss_fn.pos_weight = loss_fn.pos_weight.to(accelerator.device)

    history = []
    best_metric = -float("inf")
    best_epoch = -1
    best_ckpt_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, accelerator, args.loss_type)
        val_metrics = evaluate(model, val_loader, loss_fn, accelerator, args.loss_type) if val_loader is not None else None
        test_metrics = evaluate(model, test_loader, loss_fn, accelerator, args.loss_type)

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "time_sec": round(time.time() - t0, 2),
        }
        history.append(record)

        ref = val_metrics if val_metrics is not None else test_metrics
        current_metric = ref.get(args.metric_for_best, float("nan"))
        if not (isinstance(current_metric, float) and math.isnan(current_metric)) and current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            unwrapped = accelerator.unwrap_model(model)
            accelerator.save({"model_state_dict": unwrapped.state_dict(), "args": vars(args)}, best_ckpt_path)

        if accelerator.is_main_process:
            def fmt(m):
                if m is None:
                    return "None"
                return f"loss={m['loss']:.4f}, acc={m['acc']:.4f}, f1={m['f1']:.4f}, roc_auc={m['roc_auc']:.4f}, pred_ones={m['preds_one']:.2f}, n={m['n']}"
            print(f"\nEpoch {epoch}/{args.epochs}  time={record['time_sec']}s")
            print("  train:", fmt(train_metrics))
            if val_metrics is not None:
                print("  val  :", fmt(val_metrics))
            print("  test :", fmt(test_metrics))
            print(f"  best_{args.metric_for_best}={best_metric:.4f} @ epoch {best_epoch}")

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_metric_name": args.metric_for_best,
                    "best_metric_value": best_metric,
                    "final_train": history[-1]["train"],
                    "final_val": history[-1]["val"],
                    "final_test": history[-1]["test"],
                    "best_model_path": str(best_ckpt_path),
                },
                f,
                indent=2,
            )

if __name__ == "__main__":
    main()