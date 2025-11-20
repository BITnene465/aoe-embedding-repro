"""Training entry points for AoE models."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from aoe.data import load_gis_splits, load_nli_dataset, load_stsb_splits
from aoe.loss import aoe_total_loss, supervised_contrastive_loss
from aoe.model import SentenceEncoder


class TextPairDataset(Dataset):
    """Tiny dataset wrapper that stores paired sentences and integer labels."""

    def __init__(self, pairs: List[Tuple[str, str, int]]) -> None:
        if not pairs:
            raise ValueError("Dataset is empty; cannot build dataloader")
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        return self._pairs[index]


def _collate_batch(batch: List[Tuple[str, str, int]]) -> Tuple[List[str], List[str], torch.LongTensor]:
    """Stack text pairs into lists and labels into a tensor."""

    texts1, texts2, labels = zip(*batch)
    return list(texts1), list(texts2), torch.tensor(labels, dtype=torch.long)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _nli_pairs(split: str, cache_dir: str | None) -> List[Tuple[str, str, int]]:
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


def _stsb_pairs(split: str, cache_dir: str | None) -> List[Tuple[str, str, int]]:
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


def _gis_pairs(split: str, cache_dir: str | None) -> List[Tuple[str, str, int]]:
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


def build_dataloader(
    task: str,
    split: str,
    batch_size: int,
    cache_dir: str | None,
) -> DataLoader:
    """Construct a dataloader that yields paired texts and discrete labels."""

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
) -> Tuple[float, float, float]:
    """Run one training epoch and return average (angle, contrastive, total) losses."""

    encoder.train()
    angle_total = 0.0
    contrast_total = 0.0
    loss_total = 0.0
    steps = 0

    for texts1, texts2, labels in dataloader:
        steps += 1
        optimizer.zero_grad()
        labels = labels.to(device)

        if method == "baseline":
            z1 = encoder.encode(texts1, device=device, max_length=max_length)
            z2 = encoder.encode(texts2, device=device, max_length=max_length)
            embeddings = torch.cat([z1, z2], dim=0)
            labels_cat = torch.cat([labels, labels], dim=0)
            cl_loss = supervised_contrastive_loss(embeddings, labels_cat, tau_cl)
            loss = cl_loss
            angle_val = 0.0
            contrast_val = cl_loss.item()
        elif method == "aoe":
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
            angle_val = stats["angle_loss"]
            contrast_val = stats["contrastive_loss"]
        else:
            raise ValueError("method must be 'baseline' or 'aoe'")

        loss.backward()
        optimizer.step()

        angle_total += angle_val
        contrast_total += contrast_val
        loss_total += loss.item()

    denom = max(steps, 1)
    return angle_total / denom, contrast_total / denom, loss_total / denom


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
	parser.add_argument("--output_dir", default="output/ckpt/default")
	parser.add_argument("--data_cache", default="data")
	parser.add_argument("--model_cache", default="models")
	parser.add_argument("--seed", type=int, default=42)

	args = parser.parse_args()

	set_seed(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
		)
		print(
			f"Epoch {epoch}: angle={angle_avg:.4f} contrast={contrast_avg:.4f} total={total_avg:.4f}",
			flush=True,
		)

	save_checkpoint(encoder, args.output_dir, args.backbone)


if __name__ == "__main__":
    main()
