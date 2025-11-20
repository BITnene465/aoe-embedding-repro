"""Utility helpers for AoE training scripts."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
    on_batch_end: Optional[Callable[[dict], None]] = None,
) -> Tuple[float, float, float, int]:
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

        if on_batch_end is not None:
            on_batch_end(
                {
                    "epoch": epoch_idx,
                    "batch": steps,
                    "train_angle": angle_val,
                    "train_contrast": contrast_val,
                    "train_total": loss.item(),
                }
            )

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                angle=f"{angle_val:.4f}",
                contrast=f"{contrast_val:.4f}",
            )

    if show_progress and hasattr(iterator, "close"):
        iterator.close()

    denom = max(steps, 1)
    return angle_total / denom, contrast_total / denom, loss_total / denom, steps


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

    for texts1, texts2, labels in iterator:
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

        if on_batch_end is not None:
            on_batch_end(
                {
                    "batch": steps,
                    "eval_angle": angle_val,
                    "eval_contrast": contrast_val,
                    "eval_total": loss.item(),
                }
            )

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                angle=f"{angle_val:.4f}",
                contrast=f"{contrast_val:.4f}",
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
    task: str  # 训练任务标识，例如 "nli"、"stsb" 或 "gis"
    method: str  # 训练策略，例如 "baseline"（纯对比）或 "aoe"（角度+对比）
    backbone: str  # 预训练模型名称，例如 "bert-base-uncased"
    batch_size: int  # 每次迭代的样本批量，例如 256
    epochs: int  # 训练轮数，例如 3
    lr: float  # 学习率，例如 2e-5
    max_length: int  # 输入文本截断长度，例如 128
    temperature_cl: float  # 对比损失温度系数，例如 0.05
    temperature_angle: float  # 角度损失温度系数，例如 0.05
    w_cl: float  # 对比损失权重，例如 1.0
    w_angle: float  # 角度损失权重，例如 1.0
    output_dir: str  # 运行结果根目录，例如 "output"
    run_name: str  # 运行名称，用于区分实验，例如 "bert_nli_aoe"
    data_cache: Optional[str]  # 数据缓存目录，例如 "data"；可为 None 使用默认值
    model_cache: Optional[str]  # 模型缓存目录，例如 "models"；可为 None
    seed: int  # 随机种子，例如 42
    eval_split: Optional[str]  # 验证集名称，例如 "validation" 或 "none"
    eval_batch_size: Optional[int]  # 验证批量大小，例如 512；None 表示沿用训练批量
    metrics_path: Optional[str]  # 自定义指标文件路径，例如 "output/logs/nli.jsonl" 或 "none"
    tensorboard_dir: Optional[str]  # TensorBoard 目录，例如 "output/tensorboard" 或 "none"
    no_progress_bar: bool  # 是否禁用 tqdm 进度条，例如 True 表示关闭

    @classmethod
    def from_args(cls, args: object) -> "TrainConfig":
        return cls(
            task=getattr(args, "task"),
            method=getattr(args, "method"),
            backbone=getattr(args, "backbone"),
            batch_size=getattr(args, "batch_size"),
            epochs=getattr(args, "epochs"),
            lr=getattr(args, "lr"),
            max_length=getattr(args, "max_length"),
            temperature_cl=getattr(args, "temperature_cl"),
            temperature_angle=getattr(args, "temperature_angle"),
            w_cl=getattr(args, "w_cl"),
            w_angle=getattr(args, "w_angle"),
            output_dir=getattr(args, "output_dir"),
            run_name=getattr(args, "run_name", "default"),
            data_cache=getattr(args, "data_cache", None),
            model_cache=getattr(args, "model_cache", None),
            seed=getattr(args, "seed", 42),
            eval_split=getattr(args, "eval_split", None),
            eval_batch_size=getattr(args, "eval_batch_size", None),
            metrics_path=getattr(args, "metrics_path", None),
            tensorboard_dir=getattr(args, "tensorboard_dir", None),
            no_progress_bar=getattr(args, "no_progress_bar", False),
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
    "TextPairDataset",
    "set_seed",
    "build_dataloader",
    "train_epoch",
    "evaluate_epoch",
    "resolve_metrics_path",
    "append_metrics",
    "resolve_tensorboard_dir",
    "TrainConfig",
]
