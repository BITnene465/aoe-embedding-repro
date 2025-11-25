"""Training entry point for AoE models."""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from aoe.data import Prompts
from aoe.model import SentenceEncoder
from aoe.train_utils import (
    TrainConfig,
    append_metrics,
    build_angle_dataloader,
    evaluate_epoch,
    resolve_metrics_path,
    resolve_tensorboard_dir,
    set_seed,
    train_epoch,
)


def save_checkpoint(encoder: SentenceEncoder, ckpt_dir: str) -> None:
    """Persist the full SentenceEncoder object for direct torch.load usage."""

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "encoder.pt")
    original_device = next(encoder.parameters()).device
    encoder.to("cpu")
    torch.save(encoder, ckpt_path)
    encoder.to(original_device)


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
    parser = argparse.ArgumentParser(description="Train AoE sentence encoders")
    parser.add_argument(
        "--dataset",
        default="stsb",
        help="Dataset spec (e.g., 'stsb', 'nli', or comma-separated like 'stsb@train,gis@train')",
    )
    parser.add_argument("--train_split", default="train")
    parser.add_argument(
        "--eval_split",
        default="validation",
        help="Validation split name; use 'none' to skip evaluation",
    )
    parser.add_argument("--backbone", default="bert-base-uncased")
    parser.add_argument("--pooling", choices=["cls", "mean", "cls_avg", "max"], default="cls")
    parser.add_argument("--prompt", default=None, help="Prompt template name (e.g., A, B, C) or custom string")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--angle_tau", type=float, default=20.0)
    parser.add_argument("--cl_scale", type=float, default=20.0)
    parser.add_argument("--w_angle", type=float, default=0.02)
    parser.add_argument("--w_cl", type=float, default=1.0)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--run_name", default="default")
    parser.add_argument("--data_cache", default="data")
    parser.add_argument("--model_cache", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        help="Optional checkpoint directory or file to initialize encoder weights",
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
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps to emulate larger batch sizes",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of optimizer steps for linear LR warmup",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = TrainConfig.from_args(args)

    set_seed(config.seed)
    
    # Initialize Accelerator
    from accelerate import Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=config.grad_accum_steps)
    device = accelerator.device

    # Only main process should create directories
    if accelerator.is_main_process:
        run_dirs = _prepare_run_dirs(config.output_dir, config.run_name)
        config.save_json(run_dirs["config"])
    else:
        # Other processes need run_dirs for logging paths, but shouldn't create them
        # We can just reconstruct the paths without makedirs
        run_dir = os.path.join(config.output_dir, config.run_name or "default")
        run_dirs = {
            "run": run_dir,
            "ckpt": os.path.join(run_dir, "ckpt"),
            "tensorboard_default": os.path.join(run_dir, "tensorboard"),
            "metrics_default": os.path.join(run_dir, "metrics.jsonl"),
            "config": os.path.join(run_dir, "train_config.json"),
        }

    # Resolve prompt
    prompt_text = None
    if config.prompt:
        if hasattr(Prompts, config.prompt):
            prompt_text = getattr(Prompts, config.prompt)
        else:
            prompt_text = config.prompt

    encoder = SentenceEncoder(
        model_name=config.backbone,
        complex_mode=True,
        pooling=config.pooling,
        cache_dir=config.model_cache,
        prompt=prompt_text,
    )

    if config.init_checkpoint:
        ckpt_path = config.init_checkpoint
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "encoder.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Init checkpoint '{ckpt_path}' is missing")
        loaded: SentenceEncoder = torch.load(ckpt_path, map_location="cpu")
        encoder.load_state_dict(loaded.state_dict())

    # No manual .to(device), handled by accelerator.prepare
    # encoder = encoder.to(device)

    train_loader = build_angle_dataloader(
        dataset=config.dataset,
        split=config.train_split,
        batch_size=config.batch_size,
        cache_dir=config.data_cache,
        shuffle=True,
        tokenizer=encoder.tokenizer,
        prompt=prompt_text,
        max_length=config.max_length,
    )

    optimizer = AdamW(encoder.parameters(), lr=config.lr)
    scheduler = None
    if config.warmup_steps > 0:
        def lr_lambda(current_step: int) -> float:
            if current_step >= config.warmup_steps:
                return 1.0
            return float(current_step + 1) / float(config.warmup_steps)

        scheduler = LambdaLR(optimizer, lr_lambda)

    eval_loader: DataLoader | None = None
    eval_split = (config.eval_split or "").lower()
    if eval_split and eval_split != "none":
        eval_loader = build_angle_dataloader(
            dataset=config.dataset,
            split=config.eval_split,
            batch_size=config.eval_batch_size or config.batch_size,
            cache_dir=config.data_cache,
            shuffle=False,
            tokenizer=encoder.tokenizer,
            prompt=prompt_text,
            max_length=config.max_length,
        )

    # Prepare everything with accelerator
    encoder, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        encoder, optimizer, train_loader, eval_loader, scheduler
    )

    metrics_path = resolve_metrics_path(run_dirs["metrics_default"], config.metrics_path)
    batch_logger = None
    global_step = 0
    
    # Only main process logs metrics
    if accelerator.is_main_process and metrics_path is not None:
        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as handle:
            handle.write("")

        def batch_logger(payload: dict) -> None:
            nonlocal global_step
            global_step += 1
            record = {
                "type": "train_batch",
                "global_step": global_step,
                "run_dir": run_dirs["run"],
            }
            record.update(payload)
            append_metrics(metrics_path, record)

    writer: SummaryWriter | None = None
    tb_dir = resolve_tensorboard_dir(run_dirs["tensorboard_default"], config.tensorboard_dir)
    # Only main process logs to tensorboard
    if accelerator.is_main_process and tb_dir is not None:
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_text("train_config/json", json.dumps(config.to_dict(), indent=2), 0)

    eval_angle = eval_contrast = eval_total = None
    angle_avg = contrast_avg = total_avg = 0.0

    for epoch in range(1, config.epochs + 1):
        angle_avg, contrast_avg, total_avg, _ = train_epoch(
            encoder,
            train_loader,
            optimizer,
            accelerator, # Pass accelerator instead of device
            angle_tau=config.angle_tau,
            cl_scale=config.cl_scale,
            w_angle=config.w_angle,
            w_cl=config.w_cl,
            max_length=config.max_length,
            epoch_idx=epoch,
            total_epochs=config.epochs,
            show_progress=(not config.no_progress_bar) and accelerator.is_main_process,
            on_batch_end=batch_logger,
            grad_accum_steps=config.grad_accum_steps,
            scheduler_step=scheduler.step if scheduler is not None else None,
        )

        eval_angle = eval_contrast = eval_total = None
        if eval_loader is not None:
            eval_angle, eval_contrast, eval_total, _ = evaluate_epoch(
                encoder,
                eval_loader,
                device, # eval can still use device, or we can update it too. Let's keep device for now as it's just a property
                angle_tau=config.angle_tau,
                cl_scale=config.cl_scale,
                w_angle=config.w_angle,
                w_cl=config.w_cl,
                max_length=config.max_length,
                show_progress=(not config.no_progress_bar) and accelerator.is_main_process,
            )

        # Logging only on main process
        if accelerator.is_main_process:
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
                    "type": "train_epoch",
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

    # Save checkpoint (only main process)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_encoder = accelerator.unwrap_model(encoder)
        save_checkpoint(unwrapped_encoder, run_dirs["ckpt"])

    if accelerator.is_main_process and writer is not None:
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
