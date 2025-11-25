"""Data loading and preprocessing utilities for AoE experiments."""

import os
import random
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


class Prompts:
    """Predefined prompts for AoE tasks."""

    A = 'Summarize sentence "{text}" in one word:"'
    B = 'You can only output one word. Summarize "{text}":"'
    C = "Represent this sentence for searching relevant passages: {text}"

    @classmethod
    def list_prompts(cls):
        for key, val in cls.__dict__.items():
            if key.startswith("_") or key == "list_prompts":
                continue
            print(f"Prompts.{key}", "=", f"'{val}'")


@dataclass
class AngleDataCollator:
    """
    Collator that handles raw data, tokenizes it, and prepares batches.
    Matches official AnglE implementation.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = "longest"
    max_length: Optional[int] = None
    return_tensors: str = "pt"
    filter_duplicate: bool = True
    text_prompt: Optional[str] = None
    query_prompt: Optional[str] = None
    doc_prompt: Optional[str] = None
    dataset_format: Optional[str] = None

    @staticmethod
    def sample_from_list(text: Union[str, List[str]]) -> str:
        if isinstance(text, list):
            return random.choice(text)
        return text

    def __call__(self, features: List[Dict], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Auto-detect dataset format from first sample if not specified
        if self.dataset_format is None:
            sample = features[0]
            if "text1" in sample and "text2" in sample and "score" in sample:
                self.dataset_format = "A"
            elif "text1" in sample and "text2" in sample and "label" in sample:
                 self.dataset_format = "A"
            elif "query" in sample and "positive" in sample and "negative" in sample:
                self.dataset_format = "C"
            elif "query" in sample and "positive" in sample:
                self.dataset_format = "B"
            else:
                if "text1" in sample and "text2" in sample:
                     self.dataset_format = "A"
                else:
                    raise NotImplementedError("Unable to detect dataset format")

        all_texts: List[str] = []
        all_labels: List[float] = []

        for feature in features:
            texts = []
            label = -1.0

            if self.dataset_format == "A":
                text1 = self.sample_from_list(feature.get("text1", feature.get("text_1")))
                text2 = self.sample_from_list(feature.get("text2", feature.get("text_2")))
                val = feature.get("score", feature.get("label"))
                if val is not None:
                    label = float(val)

                if self.text_prompt is not None:
                    text1 = self.text_prompt.format(text=text1)
                    text2 = self.text_prompt.format(text=text2)
                texts = [text1, text2]

            elif self.dataset_format == "B":
                query = self.sample_from_list(feature["query"])
                positive = self.sample_from_list(feature["positive"])
                if self.query_prompt is not None:
                    query = self.query_prompt.format(text=query)
                if self.doc_prompt is not None:
                    positive = self.doc_prompt.format(text=positive)
                texts = [query, positive]

            elif self.dataset_format == "C":
                query = self.sample_from_list(feature["query"])
                positive = self.sample_from_list(feature["positive"])
                negative = self.sample_from_list(feature["negative"])
                if self.query_prompt is not None:
                    query = self.query_prompt.format(text=query)
                if self.doc_prompt is not None:
                    positive = self.doc_prompt.format(text=positive)
                    negative = self.doc_prompt.format(text=negative)
                texts = [query, positive, negative]

            all_texts.extend(texts)
            all_labels.extend([label] * len(texts))

        if not all_texts:
             raise ValueError("No features to process (empty input)")

        batch = self.tokenizer(
            all_texts,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            return_tensors=return_tensors,
        )

        if self.filter_duplicate:
            unique_indices = []
            seen = set()
            input_ids = batch["input_ids"]
            
            for idx, row in enumerate(input_ids):
                row_tuple = tuple(row.tolist())
                if row_tuple not in seen:
                    seen.add(row_tuple)
                    unique_indices.append(idx)
            
            if len(unique_indices) < len(all_texts):
                for key in batch:
                    batch[key] = batch[key][unique_indices]
                all_labels = [all_labels[i] for i in unique_indices]

        batch["labels"] = torch.tensor(all_labels, dtype=torch.float)
        return batch


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


def load_nli_dataset(split: str = "train", cache_dir: str = "data") -> Dataset:
    """Load SNLI and MultiNLI from local disk."""
    snli_path = os.path.join(cache_dir, "snli")
    mnli_path = os.path.join(cache_dir, "multi_nli")

    if not os.path.exists(snli_path) or not os.path.exists(mnli_path):
        raise FileNotFoundError(f"NLI datasets not found in {cache_dir}. Please run download scripts first.")

    snli = load_from_disk(snli_path)
    if split in snli:
        snli = snli[split]
    snli = snli.filter(_filter_valid_labels)

    mnli = load_from_disk(mnli_path)
    mnli_split = _resolve_mnli_split(split)
    if mnli_split in mnli:
        mnli = mnli[mnli_split]
    mnli = mnli.filter(_filter_valid_labels)

    required_cols = {"premise", "hypothesis", "label"}
    if not required_cols.issubset(snli.column_names):
        raise ValueError("SNLI split is missing required fields")
    if not required_cols.issubset(mnli.column_names):
        raise ValueError("MultiNLI split is missing required fields")

    combined = concatenate_datasets([snli, mnli])
    return combined


def load_stsb_splits(cache_dir: str = "data") -> Dict[str, Dataset]:
    """Return STS-B splits from local disk."""
    stsb_path = os.path.join(cache_dir, "stsb")
    if not os.path.exists(stsb_path):
        raise FileNotFoundError(f"STS-B dataset not found in {cache_dir}. Please run download scripts first.")

    ds_dict = load_from_disk(stsb_path)
    splits: Dict[str, Dataset] = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in ds_dict:
            continue
        split_ds = ds_dict[split_name]
        if "label" in split_ds.column_names and "score" not in split_ds.column_names:
            split_ds = split_ds.rename_column("label", "score")
        splits[split_name] = split_ds
    return splits


def load_sickr_split(cache_dir: str = "data") -> Dataset:
    """Load the SICK-R dataset from local disk."""
    sickr_path = os.path.join(cache_dir, "sickr")
    if not os.path.exists(sickr_path):
        raise FileNotFoundError(f"SICK-R dataset not found in {cache_dir}. Please run download scripts first.")

    ds = load_from_disk(sickr_path)
    # sickr usually only has 'test' split in some versions, or 'train'/'validation'/'test'
    # The original code hardcoded split="test" for mteb/sickr-sts
    if "test" in ds:
        ds = ds["test"]
    
    rename_map = {
        "sentence1": "text1",
        "sentence2": "text2",
        "sentence_A": "text1",
        "sentence_B": "text2",
        "relatedness_score": "score",
    }
    for src, tgt in rename_map.items():
        if src in ds.column_names and tgt not in ds.column_names:
            ds = ds.rename_column(src, tgt)
    return ds


def load_gis_splits(cache_dir: str = "data") -> Dict[str, Dataset]:
    """Load the GitHub Issue Similarity dataset from local disk."""
    gis_path = os.path.join(cache_dir, "gis")
    if not os.path.exists(gis_path):
        raise FileNotFoundError(f"GIS dataset not found in {cache_dir}. Please run download scripts first.")

    ds_dict = load_from_disk(gis_path)

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
            "text1": text1,
            "text2": text2,
            "score": label,
        }

    splits: Dict[str, Dataset] = {}
    # ds_dict from load_from_disk might be a DatasetDict or just a Dataset depending on how it was saved.
    # Usually load_dataset returns DatasetDict which saves as such.
    if hasattr(ds_dict, "items"):
        for split_name, split_ds in ds_dict.items():
            splits[split_name] = split_ds.map(
                normalize,
                remove_columns=split_ds.column_names,
            )
    else:
        # Fallback if it's a single dataset (unlikely for GIS but good safety)
        splits["train"] = ds_dict.map(normalize, remove_columns=ds_dict.column_names)
        
    return splits


def _nli_to_angle_pairs(dataset: Dataset) -> List[Dict[str, object]]:
    label_scores = {
        "entailment": 1.0,
        "contradiction": 0.0,
        0: 1.0,
        2: 0.0,
    }
    pairs: List[Dict[str, object]] = []
    for example in dataset:
        premise = example.get("premise")
        hypothesis = example.get("hypothesis")
        label = example.get("label")
        if premise is None or hypothesis is None or label is None:
            continue
        if label == "neutral" or label == 1:
            continue
        score = label_scores.get(label)
        if score is None:
            continue
        pairs.append(
            {
                "text1": str(premise),
                "text2": str(hypothesis),
                "score": float(score),
            }
        )
    if not pairs:
        raise ValueError("NLI split did not yield any valid pairs")
    return pairs


def _dataset_to_angle_pairs(dataset: Dataset) -> List[Dict[str, object]]:
    pairs: List[Dict[str, object]] = []
    for example in dataset:
        sent1 = example.get("text1") or example.get("sentence1")
        sent2 = example.get("text2") or example.get("sentence2")
        score = example.get("score") or example.get("label")
        if sent1 is None or sent2 is None or score is None:
            continue
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            continue
        pairs.append(
            {
                "text1": str(sent1),
                "text2": str(sent2),
                "score": score_val,
            }
        )
    if not pairs:
        raise ValueError("Requested dataset split does not contain scored examples")
    return pairs


def load_angle_pairs(dataset: str, split: str, cache_dir: str = "data") -> List[Dict[str, object]]:
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


def _load_single_dataset(name: str, split: str, cache_dir: str) -> List[Dict[str, object]]:
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


