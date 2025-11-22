"""Data loading and preprocessing utilities for AoE experiments."""

import random
from typing import Dict, List, Optional

from datasets import Dataset, concatenate_datasets, load_dataset


def _pick_field(example: Dict[str, object], candidates: tuple[str, ...], default: object) -> object:
    """Return the first non-None field value from the provided candidate keys."""

    for key in candidates:
        if key in example and example[key] is not None:
            return example[key]
    return default


def _resolve_mnli_split(split: str) -> str:
    """Map user-friendly split names to the MultiNLI equivalents."""

    mapping = {
        "validation": "validation_matched",
        "val": "validation_matched",
        "dev": "validation_matched",
        "test": "test_matched",
    }
    return mapping.get(split, split)


def _filter_valid_labels(example: Dict[str, int]) -> bool:
    """Return True when the example label is a non-negative integer."""

    label = example.get("label")
    if label is None:
        return False
    if isinstance(label, bool):
        return False
    if isinstance(label, (int, float)):
        return int(label) == label and int(label) >= 0
    return False


def load_nli_dataset(split: str = "train", cache_dir: Optional[str] = "data") -> Dataset:
    """Load SNLI and MultiNLI, clean invalid labels, and merge them into one dataset."""

    snli = load_dataset("snli", split=split, cache_dir=cache_dir).filter(_filter_valid_labels)

    mnli_split = _resolve_mnli_split(split)
    mnli = load_dataset("multi_nli", split=mnli_split, cache_dir=cache_dir).filter(
        _filter_valid_labels
    )

    required_cols = {"premise", "hypothesis", "label"}
    if not required_cols.issubset(snli.column_names):
        raise ValueError("SNLI split is missing required fields")
    if not required_cols.issubset(mnli.column_names):
        raise ValueError("MultiNLI split is missing required fields")

    combined = concatenate_datasets([snli, mnli])
    return combined


def load_stsb_splits(cache_dir: Optional[str] = "data") -> Dict[str, Dataset]:
    """Return STS-B splits with fields (sentence1, sentence2, score)."""

    ds_dict = load_dataset("glue", "stsb", cache_dir=cache_dir)
    splits: Dict[str, Dataset] = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in ds_dict:
            continue
        split_ds = ds_dict[split_name]
        if "label" in split_ds.column_names and "score" not in split_ds.column_names:
            split_ds = split_ds.rename_column("label", "score")
        splits[split_name] = split_ds
    return splits


def load_sickr_split(cache_dir: Optional[str] = "data") -> Dataset:
    """Load the SICK-R dataset (validation split) with normalized field names."""

    try:
        ds = load_dataset(
            "sick",
            split="validation",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception as exc:  # pragma: no cover - dataset availability guard
        raise NotImplementedError(
            "SICK-R is not available via datasets in this environment."
        ) from exc

    rename_map = {
        "sentence_A": "sentence1",
        "sentence_B": "sentence2",
        "relatedness_score": "score",
    }
    for src, tgt in rename_map.items():
        if src in ds.column_names and tgt not in ds.column_names:
            ds = ds.rename_column(src, tgt)
    return ds


def load_gis_splits(cache_dir: Optional[str] = "data") -> Dict[str, Dataset]:
    """Load the GitHub Issue Similarity dataset and expose sentence fields plus scores."""

    ds_dict = load_dataset("WhereIsAI/github-issue-similarity", cache_dir=cache_dir)

    def normalize(example: Dict[str, object]) -> Dict[str, object]:
        text1 = _pick_field(
            example,
            ("issue1", "issue_1", "text1", "text_1", "body_1", "title_1"),
            "",
        )
        text2 = _pick_field(
            example,
            ("issue2", "issue_2", "text2", "text_2", "body_2", "title_2"),
            "",
        )
        label = _pick_field(
            example,
            ("score", "label", "similar", "similarity"),
            0.0,
        )
        text1 = str(text1)
        text2 = str(text2)
        if isinstance(label, bool):
            label = float(label)
        if isinstance(label, int):
            label = float(label)
        if not isinstance(label, float):
            try:
                label = float(label)
            except (TypeError, ValueError):
                label = 0.0
        return {
            "sentence1": text1,
            "sentence2": text2,
            "score": label,
        }

    splits: Dict[str, Dataset] = {}
    for split_name, split_ds in ds_dict.items():
        splits[split_name] = split_ds.map(
            normalize,
            remove_columns=split_ds.column_names,
        )
    return splits


def _nli_to_angle_pairs(dataset: Dataset) -> List[Dict[str, object]]:
    label_scores = {
        "entailment": 1.0,
        "neutral": 0.5,
        "contradiction": 0.0,
        0: 1.0,
        1: 0.5,
        2: 0.0,
    }
    pairs: List[Dict[str, object]] = []
    for example in dataset:
        premise = example.get("premise")
        hypothesis = example.get("hypothesis")
        label = example.get("label")
        if premise is None or hypothesis is None or label is None:
            continue
        score = label_scores.get(label)
        if score is None:
            continue
        pairs.append(
            {
                "sentence1": str(premise),
                "sentence2": str(hypothesis),
                "score": float(score),
            }
        )
    if not pairs:
        raise ValueError("NLI split did not yield any valid pairs")
    return pairs


def _dataset_to_angle_pairs(dataset: Dataset) -> List[Dict[str, object]]:
    pairs: List[Dict[str, object]] = []
    for example in dataset:
        sent1 = example.get("sentence1")
        sent2 = example.get("sentence2")
        score = example.get("score")
        if sent1 is None or sent2 is None or score is None:
            continue
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            continue
        pairs.append(
            {
                "sentence1": str(sent1),
                "sentence2": str(sent2),
                "score": score_val,
            }
        )
    if not pairs:
        raise ValueError("Requested dataset split does not contain scored examples")
    return pairs


def load_angle_pairs(dataset: str, split: str, cache_dir: Optional[str] = "data") -> List[Dict[str, object]]:
    dataset_raw = (dataset or "").strip()
    split_norm = (split or "").lower()
    if not dataset_raw:
        raise ValueError("Dataset name must be provided")

    tokens = [part.strip() for part in dataset_raw.replace("+", ",").split(",") if part.strip()]
    if not tokens:
        raise ValueError("Dataset string did not contain any valid entries")

    if len(tokens) > 1:
        merged: List[Dict[str, object]] = []
        for token in tokens:
            name, custom_split = _parse_dataset_token(token)
            use_split = custom_split or split_norm or "train"
            merged.extend(_load_single_dataset(name, use_split, cache_dir))
        if not merged:
            raise ValueError("Combined dataset spec produced no valid angle pairs")
        random.shuffle(merged)
        return merged

    name, custom_split = _parse_dataset_token(tokens[0])
    use_split = custom_split or split_norm or "train"
    return _load_single_dataset(name, use_split, cache_dir)


def _parse_dataset_token(token: str) -> tuple[str, Optional[str]]:
    if "@" not in token:
        return token.lower(), None
    name, split = token.split("@", 1)
    return name.lower(), (split or "").lower() or None


def _load_single_dataset(name: str, split: str, cache_dir: Optional[str]) -> List[Dict[str, object]]:
    split_norm = (split or "").lower()

    if name == "nli":
        raw = load_nli_dataset(split_norm or "train", cache_dir=cache_dir)
        return _nli_to_angle_pairs(raw)

    if name == "stsb":
        splits = load_stsb_splits(cache_dir=cache_dir)
        if split_norm not in splits:
            raise ValueError(f"STS-B split '{split}' is unavailable")
        return _dataset_to_angle_pairs(splits[split_norm])

    if name == "gis":
        splits = load_gis_splits(cache_dir=cache_dir)
        if split_norm not in splits:
            raise ValueError(f"GIS split '{split}' is unavailable")
        return _dataset_to_angle_pairs(splits[split_norm])

    if name == "sickr":
        if split_norm not in {"validation", "val", "dev", "test"}:
            raise ValueError("SICK-R exposes only the validation split for scored data")
        return _dataset_to_angle_pairs(load_sickr_split(cache_dir=cache_dir))

    raise ValueError(f"Unknown dataset '{name}' for AoE angle training")
