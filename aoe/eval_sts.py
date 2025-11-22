"""Evaluation routines for running AoE checkpoints on STS datasets."""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from aoe.data import load_gis_splits, load_sickr_split, load_stsb_splits
from aoe.model import SentenceEncoder


def load_encoder_from_ckpt(ckpt: str, model_cache: str | None = None) -> SentenceEncoder:
    """Instantiate a SentenceEncoder from a full-object checkpoint."""

    ckpt_path = os.path.join(ckpt, "encoder.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Expected checkpoint at {ckpt_path}")

    encoder = torch.load(ckpt_path, map_location="cpu")
    encoder.eval()
    return encoder


def _encode_texts(
    encoder: SentenceEncoder,
    texts: List[str],
    device: torch.device,
    max_length: int,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode a list of texts into real-valued sentence embeddings."""

    chunks = []
    iterator = range(0, len(texts), batch_size)
    for start in tqdm(iterator, desc="Encoding", leave=False):
        batch = texts[start : start + batch_size]
        encoded = encoder.encode(batch, device=device, max_length=max_length)
        if isinstance(encoded, tuple):
            encoded = torch.cat(encoded, dim=-1)
        chunks.append(encoded.detach().cpu())
    return torch.cat(chunks, dim=0)


def _prepare_dataset(dataset_name: str, cache_dir: str | None, stsb_split: str = "validation"):
    """Return sentence pairs and scores for the requested dataset name."""

    if dataset_name == "stsb":
        splits = load_stsb_splits(cache_dir=cache_dir)
        split_key = (stsb_split or "validation").lower()
        dataset = splits.get(split_key)
        if dataset is None:
            raise ValueError(f"STS-B split '{split_key}' unavailable")
    elif dataset_name == "gis":
        splits = load_gis_splits(cache_dir=cache_dir)
        dataset = splits.get("test") or splits.get("validation") or splits.get("train")
        if dataset is None:
            raise ValueError("GIS dataset does not expose usable splits")
    elif dataset_name == "sickr":
        dataset = load_sickr_split(cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    s1, s2, scores = [], [], []
    for example in dataset:
        sent1 = example.get("text1")
        sent2 = example.get("text2")
        score = example.get("score")
        if sent1 is None or sent2 is None or score is None:
            continue
        score_val = float(score)
        if dataset_name == "stsb" and score_val < 0:
            continue
        s1.append(str(sent1))
        s2.append(str(sent2))
        scores.append(score_val)

    if not s1:
        raise ValueError(f"Dataset '{dataset_name}' produced zero scored examples")
    return s1, s2, np.asarray(scores, dtype=np.float32)


def eval_dataset(
    encoder: SentenceEncoder,
    dataset_name: str,
    device: torch.device,
    max_length: int,
    data_cache: str | None,
    stsb_split: str = "validation",
) -> float:
    """Evaluate a checkpoint on a dataset and return Spearman correlation."""

    encoder.eval()
    sentences1, sentences2, gold_scores = _prepare_dataset(
        dataset_name,
        data_cache,
        stsb_split=stsb_split,
    )

    emb1 = _encode_texts(encoder, sentences1, device=device, max_length=max_length)
    emb2 = _encode_texts(encoder, sentences2, device=device, max_length=max_length)

    dot = (emb1 * emb2).sum(dim=1)
    norms = emb1.norm(dim=1) * emb2.norm(dim=1)
    cosine_sim = (dot / norms.clamp(min=1e-8)).numpy()

    correlation = spearmanr(cosine_sim, gold_scores).correlation
    if correlation is None or np.isnan(correlation):
        raise ValueError(f"Spearman correlation undefined for dataset '{dataset_name}'")
    return float(correlation)


def main() -> None:
    """CLI entry point for STS-style evaluation using saved checkpoints."""
    parser = argparse.ArgumentParser(description="Evaluate AoE checkpoints on STS datasets")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint directory")
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated list of datasets (stsb,gis,sickr,sts_all)",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--data_cache", default="data")
    parser.add_argument("--model_cache", default="models")
    parser.add_argument(
        "--stsb_split",
        default="validation",
        help="STS-B split to evaluate (validation by default since test lacks labels)",
    )
    args = parser.parse_args()

    encoder = load_encoder_from_ckpt(args.ckpt, model_cache=args.model_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    requested = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    if not requested:
        raise ValueError("At least one dataset must be specified")

    if "sts_all" in requested:
        requested = [name for name in requested if name != "sts_all"]
        for candidate in ["stsb", "gis", "sickr"]:
            if candidate not in requested:
                requested.append(candidate)

    results = {}
    for dataset_name in tqdm(requested, desc="Datasets"):
        try:
            score = eval_dataset(
                encoder,
                dataset_name,
                device=device,
                max_length=args.max_length,
                data_cache=args.data_cache,
                stsb_split=args.stsb_split,
            )
        except NotImplementedError:
            print(f"Dataset '{dataset_name}' is not implemented; skipping.")
            continue
        except ValueError as exc:
            print(f"Skipping dataset '{dataset_name}': {exc}")
            continue

        results[dataset_name] = score
        print(f"Dataset: {dataset_name}, Spearman: {score:.4f}")

    if results:
        avg = sum(results.values()) / len(results)
        print(f"Average Spearman: {avg:.4f}")
    else:
        print("No datasets were evaluated successfully.")


if __name__ == "__main__":
    main()
