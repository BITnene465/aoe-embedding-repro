"""Analysis helpers for inspecting AoE embedding behaviors."""

from __future__ import annotations

import argparse
import math
import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from aoe.data import load_nli_dataset
from aoe.model import SentenceEncoder


def _encode_texts(
    encoder: SentenceEncoder,
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int = 256,
) -> torch.Tensor:
    """Encode texts in mini-batches and return stacked embeddings."""

    outputs = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            emb = encoder.encode(batch, device=device, max_length=max_length)
            if isinstance(emb, tuple):
                emb = torch.cat(emb, dim=-1)
            outputs.append(emb.detach().cpu())
    return torch.cat(outputs, dim=0)


def _sample_nli_pairs(max_samples: int, cache_dir: str | None) -> Tuple[list[str], list[str]]:
    """Randomly sample premise-hypothesis pairs from the NLI corpus."""

    dataset = load_nli_dataset("train", cache_dir=cache_dir)
    total = len(dataset)
    sample_size = min(max_samples, total)
    indices = (
        random.sample(range(total), sample_size)
        if sample_size < total
        else list(range(total))
    )

    premises: list[str] = []
    hypotheses: list[str] = []
    for idx in indices:
        example = dataset[int(idx)]
        premise = example.get("premise")
        hypothesis = example.get("hypothesis")
        if premise is None or hypothesis is None:
            continue
        premises.append(str(premise))
        hypotheses.append(str(hypothesis))

    if not premises:
        raise ValueError("No valid NLI pairs collected for cosine analysis")
    return premises, hypotheses


def cosine_saturation(
	backbone: str,
	max_samples: int,
	max_length: int,
	model_cache: str | None,
	data_cache: str | None,
	plot_dir: str,
) -> None:
	"""Compute and visualize cosine saturation on NLI sentence pairs."""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	encoder = SentenceEncoder(
		model_name=backbone,
		complex_mode=False,
		cache_dir=model_cache,
	).to(device)

	premises, hypotheses = _sample_nli_pairs(max_samples, cache_dir=data_cache)
	prem_emb = _encode_texts(encoder, premises, device=device, max_length=max_length)
	hyp_emb = _encode_texts(encoder, hypotheses, device=device, max_length=max_length)

	dot = (prem_emb * hyp_emb).sum(dim=1)
	norms = prem_emb.norm(dim=1) * hyp_emb.norm(dim=1)
	cosine = (dot / norms.clamp(min=1e-9)).numpy()

	pct_95 = 100.0 * float(np.mean(cosine > 0.95))
	pct_80 = 100.0 * float(np.mean(cosine > 0.80))
	if not math.isfinite(pct_95):
		pct_95 = 0.0
	if not math.isfinite(pct_80):
		pct_80 = 0.0
	print(f"cos > 0.95: {pct_95:.1f}%, cos > 0.80: {pct_80:.1f}%")

	os.makedirs(plot_dir, exist_ok=True)
	plot_path = os.path.join(plot_dir, "cosine_hist.png")
	plt.figure(figsize=(6, 4))
	plt.hist(cosine, bins=50, range=(-0.2, 1.0), color="steelblue", edgecolor="black")
	plt.xlabel("Cosine similarity (premise vs hypothesis)")
	plt.ylabel("Count")
	plt.title("Cosine saturation on NLI pairs")
	plt.tight_layout()
	plt.savefig(plot_path, dpi=200)
	plt.close()
	print(f"Saved histogram to {plot_path}")


def main() -> None:
	"""CLI entry point for AoE analysis utilities."""

	parser = argparse.ArgumentParser(description="AoE analysis tools")
	parser.add_argument("--mode", required=True, choices=["cosine_saturation"])
	parser.add_argument("--backbone", default="bert-base-uncased")
	parser.add_argument("--max_samples", type=int, default=50_000)
	parser.add_argument("--max_length", type=int, default=64)
	parser.add_argument("--data_cache", default="data")
	parser.add_argument("--model_cache", default="models")
	parser.add_argument("--plot_dir", default="output/plot")
	args = parser.parse_args()

	if args.mode == "cosine_saturation":
		cosine_saturation(
			args.backbone,
			args.max_samples,
			args.max_length,
			model_cache=args.model_cache,
			data_cache=args.data_cache,
			plot_dir=args.plot_dir,
		)
	else:  # pragma: no cover - future extension guard
		raise ValueError(f"Unknown analysis mode '{args.mode}'")


if __name__ == "__main__":
	main()
