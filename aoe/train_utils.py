"""Utility helpers for AoE training scripts."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from aoe.data import load_angle_pairs
from aoe.loss import aoe_total_loss
from aoe.model import SentenceEncoder


class AnglePairDataset(Dataset):
    """Dataset of text pairs with real-valued similarity scores."""

    def __init__(self, pairs: List[Dict[str, object]]) -> None:
        if not pairs:
            raise ValueError("AoE dataset is empty; cannot build dataloader")
        self._pairs = pairs

    def __len__(self) -> int:  # pragma: no cover - trivial passthrough
        return len(self._pairs)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self._pairs[index]


def _angle_collate(batch: Sequence[Dict[str, object]]) -> Tuple[List[str], torch.Tensor]:
    texts: List[str] = []
    scores: List[float] = []
    for example in batch:
        sent1 = str(example["sentence1"])
        sent2 = str(example["sentence2"])
        score = float(example["score"])
        texts.extend([sent1, sent2])
        scores.extend([score, score])
    return texts, torch.tensor(scores, dtype=torch.float32)


def build_angle_dataloader(
    dataset: str,
    split: str,
    batch_size: int,
    cache_dir: Optional[str],
    shuffle: bool,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pairs = load_angle_pairs(dataset, split, cache_dir)
    return DataLoader(
        AnglePairDataset(pairs),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_angle_collate,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _encode_zigzag(
    encoder: SentenceEncoder,
    texts: List[str],
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    encoded = encoder.encode(texts, device=device, max_length=max_length)
    if not isinstance(encoded, tuple):
        raise ValueError("SentenceEncoder must run in complex_mode=True for AoE training")
    z_re, z_im = encoded
    return torch.cat([z_re, z_im], dim=1)


def train_epoch(
    encoder: SentenceEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    angle_tau: float,
    cl_scale: float,
    w_angle: float,
    w_cl: float,
    max_length: int,
    epoch_idx: int,
    total_epochs: int,
    show_progress: bool,
    on_batch_end: Optional[Callable[[dict], None]] = None,
    grad_accum_steps: int = 1,
    scheduler_step: Optional[Callable[[], None]] = None,
) -> Tuple[float, float, float, int]:
    encoder.train()
    grad_accum_steps = max(1, grad_accum_steps)
    angle_total = 0.0
    contrast_total = 0.0
    loss_total = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)

    iterator = dataloader
    if show_progress:
        desc = f"Epoch {epoch_idx}/{total_epochs}"
        iterator = tqdm(dataloader, desc=desc, leave=False)

    for texts, scores in iterator:
        steps += 1
        y_true = scores.to(device)
        y_pred = _encode_zigzag(encoder, texts, device, max_length)
        loss, stats = aoe_total_loss(
            y_true,
            y_pred,
            angle_tau=angle_tau,
            cl_scale=cl_scale,
            w_angle=w_angle,
            w_cl=w_cl,
        )
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()
        if steps % grad_accum_steps == 0:
            optimizer.step()
            if scheduler_step is not None:
                scheduler_step()
            optimizer.zero_grad(set_to_none=True)
        angle_total += stats["angle_loss"]
        contrast_total += stats["contrastive_loss"]
        loss_total += stats["total_loss"]

        if on_batch_end is not None:
            on_batch_end(
                {
                    "epoch": epoch_idx,
                    "batch": steps,
                    "train_angle": stats["angle_loss"],
                    "train_contrast": stats["contrastive_loss"],
                    "train_total": stats["total_loss"],
                }
            )

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{stats['total_loss']:.4f}",
                angle=f"{stats['angle_loss']:.4f}",
                contrast=f"{stats['contrastive_loss']:.4f}",
            )

    remainder = steps % grad_accum_steps
    if remainder != 0:
        optimizer.step()
        if scheduler_step is not None:
            scheduler_step()
        optimizer.zero_grad(set_to_none=True)

    if show_progress and hasattr(iterator, "close"):
        iterator.close()

    denom = max(steps, 1)
    return angle_total / denom, contrast_total / denom, loss_total / denom, steps


@torch.no_grad()
def evaluate_epoch(
    encoder: SentenceEncoder,
    dataloader: DataLoader,
    device: torch.device,
    angle_tau: float,
    cl_scale: float,
    w_angle: float,
    w_cl: float,
    max_length: int,
    show_progress: bool = False,
    on_batch_end: Optional[Callable[[dict], None]] = None,
) -> Tuple[float, float, float, int]:
    encoder.eval()
    angle_total = 0.0
    contrast_total = 0.0
    loss_total = 0.0
    steps = 0

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Eval", leave=False)

    for texts, scores in iterator:
        steps += 1
        y_true = scores.to(device)
        y_pred = _encode_zigzag(encoder, texts, device, max_length)
        loss, stats = aoe_total_loss(
            y_true,
            y_pred,
            angle_tau=angle_tau,
            cl_scale=cl_scale,
            w_angle=w_angle,
            w_cl=w_cl,
        )
        angle_total += stats["angle_loss"]
        contrast_total += stats["contrastive_loss"]
        loss_total += stats["total_loss"]

        if on_batch_end is not None:
            on_batch_end(
                {
                    "batch": steps,
                    "eval_angle": stats["angle_loss"],
                    "eval_contrast": stats["contrastive_loss"],
                    "eval_total": stats["total_loss"],
                }
            )

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{stats['total_loss']:.4f}",
                angle=f"{stats['angle_loss']:.4f}",
                contrast=f"{stats['contrastive_loss']:.4f}",
            )

    if show_progress and hasattr(iterator, "close"):
        iterator.close()

    denom = max(steps, 1)
    return angle_total / denom, contrast_total / denom, loss_total / denom, steps


def resolve_metrics_path(default_path: str, override: Optional[str]) -> Optional[str]:
    if override is None:
        return default_path
    lowered = override.lower()
    if lowered in {"none", ""}:
        return None
    return override


def append_metrics(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def resolve_tensorboard_dir(default_dir: str, override: Optional[str]) -> Optional[str]:
    if override is None:
        return default_dir
    lowered = override.lower()
    if lowered in {"none", ""}:
        return None
    return override


@dataclass
class TrainConfig:
    dataset: str
    train_split: str
    eval_split: Optional[str]
    backbone: str
    pooling: str
    batch_size: int
    eval_batch_size: Optional[int]
    epochs: int
    lr: float
    max_length: int
    angle_tau: float
    cl_scale: float
    w_angle: float
    w_cl: float
    output_dir: str
    run_name: str
    data_cache: Optional[str]
    model_cache: Optional[str]
    init_checkpoint: Optional[str]
    seed: int
    metrics_path: Optional[str]
    tensorboard_dir: Optional[str]
    no_progress_bar: bool
    grad_accum_steps: int
    warmup_steps: int

    @classmethod
    def from_args(cls, args: object) -> "TrainConfig":
        return cls(
            dataset=getattr(args, "dataset"),
            train_split=getattr(args, "train_split", "train"),
            eval_split=getattr(args, "eval_split", None),
            backbone=getattr(args, "backbone"),
            pooling=getattr(args, "pooling", "cls"),
            batch_size=getattr(args, "batch_size"),
            eval_batch_size=getattr(args, "eval_batch_size", None),
            epochs=getattr(args, "epochs"),
            lr=getattr(args, "lr"),
            max_length=getattr(args, "max_length"),
            angle_tau=getattr(args, "angle_tau"),
            cl_scale=getattr(args, "cl_scale"),
            w_angle=getattr(args, "w_angle"),
            w_cl=getattr(args, "w_cl"),
            output_dir=getattr(args, "output_dir"),
            run_name=getattr(args, "run_name", "default"),
            data_cache=getattr(args, "data_cache", None),
            model_cache=getattr(args, "model_cache", None),
            init_checkpoint=getattr(args, "init_checkpoint", None),
            seed=getattr(args, "seed", 42),
            metrics_path=getattr(args, "metrics_path", None),
            tensorboard_dir=getattr(args, "tensorboard_dir", None),
            no_progress_bar=getattr(args, "no_progress_bar", False),
            grad_accum_steps=getattr(args, "grad_accum_steps", 1),
            warmup_steps=getattr(args, "warmup_steps", 0),
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def filtered_hparams(self) -> Dict[str, object]:
        return {
            key: value
            for key, value in self.to_dict().items()
            if isinstance(value, (int, float, str, bool))
        }

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


__all__ = [
    "AnglePairDataset",
    "set_seed",
    "build_angle_dataloader",
    "train_epoch",
    "evaluate_epoch",
    "resolve_metrics_path",
    "append_metrics",
    "resolve_tensorboard_dir",
    "TrainConfig",
]
