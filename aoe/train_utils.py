"""Utility helpers for AoE training scripts."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from aoe.data import load_gis_splits, load_nli_dataset, load_stsb_splits
from aoe.loss import aoe_total_loss, supervised_contrastive_loss
from aoe.model import SentenceEncoder


class TextPairDataset(Dataset):
    """Simple dataset wrapper for paired sentences and labels."""

    def __init__(self, pairs: List[Tuple[str, str, int]]) -> None:
        if not pairs:
            raise ValueError("Dataset is empty; cannot build dataloader")
        self._pairs = pairs

    def __len__(self) -> int:  # pragma: no cover - trivial passthrough
        return len(self._pairs)

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        return self._pairs[index]


def _collate_batch(batch: List[Tuple[str, str, int]]) -> Tuple[List[str], List[str], torch.LongTensor]:
    texts1, texts2, labels = zip(*batch)
    return list(texts1), list(texts2), torch.tensor(labels, dtype=torch.long)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _nli_pairs(split: str, cache_dir: Optional[str]) -> List[Tuple[str, str, int]]:
    dataset = load_nli_dataset(split, cache_dir=cache_dir)
    pairs: List[Tuple[str, str, int]] = []
    for example in dataset:
        premise = example.get("premise")
        hypothesis = example.get("hypothesis")
        label = example.get("label")
        if premise is None or hypothesis is None or label is None:
            continue
        label_int = int(label)
        if label_int < 0:
            continue
        pairs.append((str(premise), str(hypothesis), label_int))
    if not pairs:
        raise ValueError(f"No valid NLI pairs found for split '{split}'")
    return pairs


def _stsb_pairs(split: str, cache_dir: Optional[str]) -> List[Tuple[str, str, int]]:
    splits = load_stsb_splits(cache_dir=cache_dir)
    if split not in splits:
        raise ValueError(f"STS-B split '{split}' is unavailable")
    dataset = splits[split]
    pairs: List[Tuple[str, str, int]] = []
    for example in dataset:
        sent1 = example.get("sentence1")
        sent2 = example.get("sentence2")
        score = example.get("score")
        if sent1 is None or sent2 is None or score is None:
            continue
        score_val = float(score)
        label = int(score_val > 2.5)
        pairs.append((str(sent1), str(sent2), label))
    if not pairs:
        raise ValueError(f"No valid STS-B pairs found for split '{split}'")
    return pairs


def _gis_pairs(split: str, cache_dir: Optional[str]) -> List[Tuple[str, str, int]]:
    splits = load_gis_splits(cache_dir=cache_dir)
    if split not in splits:
        raise ValueError(f"GIS split '{split}' is unavailable")
    dataset = splits[split]
    pairs: List[Tuple[str, str, int]] = []
    for example in dataset:
        sent1 = example.get("sentence1")
        sent2 = example.get("sentence2")
        score = example.get("score")
        if sent1 is None or sent2 is None or score is None:
            continue
        score_val = float(score)
        label = int(score_val > 0.5)
        pairs.append((str(sent1), str(sent2), label))
    if not pairs:
        raise ValueError(f"No valid GIS pairs found for split '{split}'")
    return pairs


def build_dataloader(task: str, split: str, batch_size: int, cache_dir: Optional[str]) -> DataLoader:
    if task == "nli":
        pairs = _nli_pairs(split, cache_dir)
    elif task == "stsb":
        pairs = _stsb_pairs(split, cache_dir)
    elif task == "gis":
        pairs = _gis_pairs(split, cache_dir)
    else:
        raise ValueError("task must be one of {'nli', 'stsb', 'gis'}")

    dataset = TextPairDataset(pairs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        collate_fn=_collate_batch,
    )


def _forward_step(
    encoder: SentenceEncoder,
    texts1: List[str],
    texts2: List[str],
    labels: torch.LongTensor,
    device: torch.device,
    method: str,
    tau_cl: float,
    tau_angle: float,
    w_cl: float,
    w_angle: float,
    max_length: int,
) -> Tuple[torch.Tensor, float, float]:
    if method == "baseline":
        z1 = encoder.encode(texts1, device=device, max_length=max_length)
        z2 = encoder.encode(texts2, device=device, max_length=max_length)
        embeddings = torch.cat([z1, z2], dim=0)
        labels_cat = torch.cat([labels, labels], dim=0)
        cl_loss = supervised_contrastive_loss(embeddings, labels_cat, tau_cl)
        return cl_loss, 0.0, cl_loss.item()

    if method == "aoe":
        z1_re, z1_im = encoder.encode(texts1, device=device, max_length=max_length)
        z2_re, z2_im = encoder.encode(texts2, device=device, max_length=max_length)
        z_re = torch.cat([z1_re, z2_re], dim=0)
        z_im = torch.cat([z1_im, z2_im], dim=0)
        labels_cat = torch.cat([labels, labels], dim=0)
        loss, stats = aoe_total_loss(
            z_re,
            z_im,
            labels_cat,
            tau_angle=tau_angle,
            tau_cl=tau_cl,
            w_angle=w_angle,
            w_cl=w_cl,
        )
        return loss, stats["angle_loss"], stats["contrastive_loss"]

    raise ValueError("method must be 'baseline' or 'aoe'")


def train_epoch(
    encoder: SentenceEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    method: str,
    tau_cl: float,
    tau_angle: float,
    w_cl: float,
    w_angle: float,
    max_length: int,
    epoch_idx: int,
    total_epochs: int,
    show_progress: bool,
) -> Tuple[float, float, float]:
    encoder.train()
    angle_total = 0.0
    contrast_total = 0.0
    loss_total = 0.0
    steps = 0

    iterator = dataloader
    if show_progress:
        desc = f"Epoch {epoch_idx}/{total_epochs}"
        iterator = tqdm(dataloader, desc=desc, leave=False)

    for texts1, texts2, labels in iterator:
        steps += 1
        optimizer.zero_grad()
        labels = labels.to(device)
        loss, angle_val, contrast_val = _forward_step(
            encoder,
            texts1,
            texts2,
            labels,
            device,
            method,
            tau_cl,
            tau_angle,
            w_cl,
            w_angle,
            max_length,
        )
        loss.backward()
        optimizer.step()
        angle_total += angle_val
        contrast_total += contrast_val
        loss_total += loss.item()

    if show_progress and hasattr(iterator, "close"):
        iterator.close()

    denom = max(steps, 1)
    return angle_total / denom, contrast_total / denom, loss_total / denom


@torch.no_grad()
def evaluate_epoch(
    encoder: SentenceEncoder,
    dataloader: DataLoader,
    device: torch.device,
    method: str,
    tau_cl: float,
    tau_angle: float,
    w_cl: float,
    w_angle: float,
    max_length: int,
) -> Tuple[float, float, float]:
    encoder.eval()
    angle_total = 0.0
    contrast_total = 0.0
    loss_total = 0.0
    steps = 0

    for texts1, texts2, labels in dataloader:
        steps += 1
        labels = labels.to(device)
        loss, angle_val, contrast_val = _forward_step(
            encoder,
            texts1,
            texts2,
            labels,
            device,
            method,
            tau_cl,
            tau_angle,
            w_cl,
            w_angle,
            max_length,
        )
        angle_total += angle_val
        contrast_total += contrast_val
        loss_total += loss.item()

    denom = max(steps, 1)
    return angle_total / denom, contrast_total / denom, loss_total / denom


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


__all__ = [
    "TextPairDataset",
    "set_seed",
    "build_dataloader",
    "train_epoch",
    "evaluate_epoch",
    "resolve_metrics_path",
    "append_metrics",
    "resolve_tensorboard_dir",
]
