"""Training entry points for AoE models."""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from aoe.model import SentenceEncoder
from aoe.train_utils import (
    append_metrics,
    build_dataloader,
    evaluate_epoch,
    resolve_metrics_path,
    resolve_tensorboard_dir,
    set_seed,
    train_epoch,
)


def save_checkpoint(
    encoder: SentenceEncoder,
    output_dir: str,
    model_name: str,
) -> None:
    """Persist model weights and minimal config to disk."""

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pt")
    config_path = os.path.join(output_dir, "config.json")

    torch.save(encoder.state_dict(), model_path)

    config = {
        "model_name": model_name,
        "complex_mode": encoder.complex_mode,
        "pooling": encoder.pooling,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _prepare_run_dirs(base_dir: str, run_name: str) -> dict[str, str]:
    run_name = run_name or "default"
    run_dir = os.path.join(base_dir, run_name)
    paths = {
        "run": run_dir,
        "ckpt": os.path.join(run_dir, "ckpt"),
        "tensorboard_default": os.path.join(run_dir, "tensorboard"),
        "metrics_default": os.path.join(run_dir, "metrics.jsonl"),
    }
    os.makedirs(paths["ckpt"], exist_ok=True)
    return paths


def main() -> None:
    """CLI entry point for training baseline or AoE sentence encoders."""

    parser = argparse.ArgumentParser(description="Train AoE or baseline sentence encoders")
    parser.add_argument("--task", choices=["nli", "stsb", "gis"], required=True)
    parser.add_argument("--method", choices=["baseline", "aoe"], default="baseline")
    parser.add_argument("--backbone", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature_cl", type=float, default=0.05)
    parser.add_argument("--temperature_angle", type=float, default=0.05)
    parser.add_argument("--w_cl", type=float, default=1.0)
    parser.add_argument("--w_angle", type=float, default=1.0)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--run_name", default="default")
    parser.add_argument("--data_cache", default="data")
    parser.add_argument("--model_cache", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument(
        "--eval_split",
        default="validation",
        help="Validation split name; use 'none' to skip evaluation",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (defaults to training batch size)",
    )
    parser.add_argument(
        "--metrics_path",
        default=None,
        help="Path to JSONL metrics log; defaults to <output_dir>/<run_name>/metrics.jsonl; use 'none' to disable",
    )
    parser.add_argument(
        "--tensorboard_dir",
        default=None,
        help="Directory for TensorBoard event files; defaults to <output_dir>/<run_name>/tensorboard; use 'none' to disable",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dirs = _prepare_run_dirs(args.output_dir, args.run_name)

    encoder = SentenceEncoder(
        model_name=args.backbone,
        complex_mode=args.method == "aoe",
        pooling="cls",
        cache_dir=args.model_cache,
    ).to(device)

    dataloader = build_dataloader(
        task=args.task,
        split="train",
        batch_size=args.batch_size,
        cache_dir=args.data_cache,
    )

    optimizer = AdamW(encoder.parameters(), lr=args.lr)

    eval_loader: DataLoader | None = None
    eval_split = args.eval_split or ""
    if eval_split and eval_split.lower() != "none":
        eval_loader = build_dataloader(
            task=args.task,
            split=eval_split,
            batch_size=args.eval_batch_size or args.batch_size,
            cache_dir=args.data_cache,
        )

    metrics_path = resolve_metrics_path(run_dirs["metrics_default"], args.metrics_path)
    if metrics_path is not None:
        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("")

    writer: SummaryWriter | None = None
    tb_dir = resolve_tensorboard_dir(run_dirs["tensorboard_default"], args.tensorboard_dir)
    if tb_dir is not None:
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_text("args/json", json.dumps(vars(args), indent=2, default=str), 0)

    eval_angle = eval_contrast = eval_total = None
    angle_avg = contrast_avg = total_avg = 0.0

    for epoch in range(1, args.epochs + 1):
        angle_avg, contrast_avg, total_avg = train_epoch(
            encoder,
            dataloader,
            optimizer,
            device,
            args.method,
            tau_cl=args.temperature_cl,
            tau_angle=args.temperature_angle,
            w_cl=args.w_cl,
            w_angle=args.w_angle,
            max_length=args.max_length,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            show_progress=not args.no_progress_bar,
        )

        eval_angle = eval_contrast = eval_total = None
        if eval_loader is not None:
            eval_angle, eval_contrast, eval_total = evaluate_epoch(
                encoder,
                eval_loader,
                device,
                args.method,
                tau_cl=args.temperature_cl,
                tau_angle=args.temperature_angle,
                w_cl=args.w_cl,
                w_angle=args.w_angle,
                max_length=args.max_length,
            )
        message = (
            f"Epoch {epoch}: train_angle={angle_avg:.4f} train_contrast={contrast_avg:.4f} "
            f"train_total={total_avg:.4f}"
        )
        if eval_total is not None:
            message += (
                f" | eval_angle={eval_angle:.4f} eval_contrast={eval_contrast:.4f}"
                f" eval_total={eval_total:.4f}"
            )
        print(message, flush=True)

        if metrics_path is not None:
            record = {
                "epoch": epoch,
                "run_dir": run_dirs["run"],
                "train_angle": angle_avg,
                "train_contrast": contrast_avg,
                "train_total": total_avg,
            }
            if eval_total is not None and eval_angle is not None and eval_contrast is not None:
                record.update(
                    {
                        "eval_angle": eval_angle,
                        "eval_contrast": eval_contrast,
                        "eval_total": eval_total,
                    }
                )
            append_metrics(metrics_path, record)

        if writer is not None:
            writer.add_scalar("loss/train_total", total_avg, epoch)
            writer.add_scalar("loss/train_angle", angle_avg, epoch)
            writer.add_scalar("loss/train_contrast", contrast_avg, epoch)
            if eval_total is not None and eval_angle is not None and eval_contrast is not None:
                writer.add_scalar("loss/eval_total", eval_total, epoch)
                writer.add_scalar("loss/eval_angle", eval_angle, epoch)
                writer.add_scalar("loss/eval_contrast", eval_contrast, epoch)

    save_checkpoint(encoder, run_dirs["ckpt"], args.backbone)

    if writer is not None:
        final_metrics = {
            "train_total": total_avg,
            "train_angle": angle_avg,
            "train_contrast": contrast_avg,
        }
        if eval_total is not None and eval_angle is not None and eval_contrast is not None:
            final_metrics.update(
                {
                    "eval_total": eval_total,
                    "eval_angle": eval_angle,
                    "eval_contrast": eval_contrast,
                }
            )
        writer.add_hparams({k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}, final_metrics)
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
