"""Training entry point for AoE models."""

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
    TrainConfig,
    append_metrics,
    build_dataloader,
    evaluate_epoch,
    resolve_metrics_path,
    resolve_tensorboard_dir,
    set_seed,
    train_epoch,
)


def save_checkpoint(encoder: SentenceEncoder, ckpt_dir: str) -> None:
    """Persist encoder weights and a minimal config to disk."""

    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    snapshot = {
        "model_name": getattr(encoder, "model_name", None),
        "complex_mode": encoder.complex_mode,
        "pooling": encoder.pooling,
    }
    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)


def _prepare_run_dirs(base_dir: str, run_name: str) -> dict[str, str]:
    run_dir = os.path.join(base_dir, run_name or "default")
    paths = {
        "run": run_dir,
        "ckpt": os.path.join(run_dir, "ckpt"),
        "tensorboard_default": os.path.join(run_dir, "tensorboard"),
        "metrics_default": os.path.join(run_dir, "metrics.jsonl"),
        "config": os.path.join(run_dir, "train_config.json"),
    }
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(paths["ckpt"], exist_ok=True)
    return paths


def _build_parser() -> argparse.ArgumentParser:
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
        help="Path to JSONL metrics log; defaults to <output>/<run>/metrics.jsonl; use 'none' to disable",
    )
    parser.add_argument(
        "--tensorboard_dir",
        default=None,
        help="Directory for TensorBoard event files; defaults to <output>/<run>/tensorboard; use 'none' to disable",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = TrainConfig.from_args(args)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dirs = _prepare_run_dirs(config.output_dir, config.run_name)
    config.save_json(run_dirs["config"])

    encoder = SentenceEncoder(
        model_name=config.backbone,
        complex_mode=config.method == "aoe",
        pooling="cls",
        cache_dir=config.model_cache,
    ).to(device)

    train_loader = build_dataloader(
        task=config.task,
        split="train",
        batch_size=config.batch_size,
        cache_dir=config.data_cache,
    )

    optimizer = AdamW(encoder.parameters(), lr=config.lr)

    eval_loader: DataLoader | None = None
    eval_split = (config.eval_split or "").lower()
    if eval_split and eval_split != "none":
        eval_loader = build_dataloader(
            task=config.task,
            split=config.eval_split,
            batch_size=config.eval_batch_size or config.batch_size,
            cache_dir=config.data_cache,
        )

    metrics_path = resolve_metrics_path(run_dirs["metrics_default"], config.metrics_path)
    if metrics_path is not None:
        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as handle:
            handle.write("")

    writer: SummaryWriter | None = None
    tb_dir = resolve_tensorboard_dir(run_dirs["tensorboard_default"], config.tensorboard_dir)
    if tb_dir is not None:
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_text("train_config/json", json.dumps(config.to_dict(), indent=2), 0)

    eval_angle = eval_contrast = eval_total = None
    angle_avg = contrast_avg = total_avg = 0.0

    for epoch in range(1, config.epochs + 1):
        angle_avg, contrast_avg, total_avg = train_epoch(
            encoder,
            train_loader,
            optimizer,
            device,
            config.method,
            tau_cl=config.temperature_cl,
            tau_angle=config.temperature_angle,
            w_cl=config.w_cl,
            w_angle=config.w_angle,
            max_length=config.max_length,
            epoch_idx=epoch,
            total_epochs=config.epochs,
            show_progress=not config.no_progress_bar,
        )

        eval_angle = eval_contrast = eval_total = None
        if eval_loader is not None:
            eval_angle, eval_contrast, eval_total = evaluate_epoch(
                encoder,
                eval_loader,
                device,
                config.method,
                tau_cl=config.temperature_cl,
                tau_angle=config.temperature_angle,
                w_cl=config.w_cl,
                w_angle=config.w_angle,
                max_length=config.max_length,
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
            if eval_total is not None:
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
            if eval_total is not None:
                writer.add_scalar("loss/eval_total", eval_total, epoch)
                writer.add_scalar("loss/eval_angle", eval_angle, epoch)
                writer.add_scalar("loss/eval_contrast", eval_contrast, epoch)

    save_checkpoint(encoder, run_dirs["ckpt"])

    if writer is not None:
        final_metrics = {
            "train_total": total_avg,
            "train_angle": angle_avg,
            "train_contrast": contrast_avg,
        }
        if eval_total is not None:
            final_metrics.update(
                {
                    "eval_total": eval_total,
                    "eval_angle": eval_angle,
                    "eval_contrast": eval_contrast,
                }
            )
        writer.add_hparams(config.filtered_hparams(), final_metrics)
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
