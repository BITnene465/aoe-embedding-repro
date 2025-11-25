"""Data loading and preprocessing utilities for AoE experiments."""

import random
from typing import Dict, List, Optional

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
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
                self.dataset_format = "A"  # Modified from 'label' to 'score' to match our internal format
            elif "text1" in sample and "text2" in sample and "label" in sample:
                 self.dataset_format = "A"
            elif "query" in sample and "positive" in sample and "negative" in sample:
                self.dataset_format = "C"
            elif "query" in sample and "positive" in sample:
                self.dataset_format = "B"
            else:
                # Fallback for our specific AnglePairDataset which uses text1, text2, score
                if "text1" in sample and "text2" in sample:
                     self.dataset_format = "A"
                else:
                    raise NotImplementedError("Unable to detect dataset format")

        # Optimization: Collect all texts first to use batch tokenization
        # This avoids the "tokenization followed by pad" warning from BertTokenizerFast
        # and significantly improves performance by reducing Python overhead.
        all_texts: List[str] = []
        all_labels: List[float] = []

        for feature in features:
            texts = []
            label = -1.0

            if self.dataset_format == "A":
                text1 = self.sample_from_list(feature.get("text1", feature.get("text_1")))
                text2 = self.sample_from_list(feature.get("text2", feature.get("text_2")))
                # Handle score/label
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

            # Collect texts and replicate label for each text in the pair/triplet
            all_texts.extend(texts)
            all_labels.extend([label] * len(texts))

        if not all_texts:
             raise ValueError("No features to process (empty input)")

        # Batch tokenize all texts at once
        # padding=self.padding (default 'longest') ensures efficient padding within the batch
        batch = self.tokenizer(
            all_texts,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            return_tensors=return_tensors,
        )

        # Filter duplicates if requested
        # We filter based on input_ids to ensure uniqueness at the token level
        if self.filter_duplicate:
            unique_indices = []
            seen = set()
            input_ids = batch["input_ids"]
            
            # Iterate and find unique indices
            for idx, row in enumerate(input_ids):
                # Convert to tuple for hashing. 
                # Note: This is necessary for set lookup.
                row_tuple = tuple(row.tolist())
                if row_tuple not in seen:
                    seen.add(row_tuple)
                    unique_indices.append(idx)
            
            # If duplicates were found, slice the batch and labels
            if len(unique_indices) < len(all_texts):
                # Slice all tensor fields in the batch (input_ids, attention_mask, etc.)
                for key in batch:
                    batch[key] = batch[key][unique_indices]
                
                # Slice labels
                all_labels = [all_labels[i] for i in unique_indices]

        # Attach labels to the batch
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
    """Return STS-B splits with fields (text1, text2, score)."""

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
    """Load the SICK-R dataset (using mteb/sickr-sts which contains all data in 'test' split)."""

    try:
        ds = load_dataset(
            "mteb/sickr-sts",
            split="test",
            cache_dir=cache_dir,
        )
    except Exception as exc:  # pragma: no cover - dataset availability guard
        raise NotImplementedError(
            "SICK-R is not available via datasets in this environment."
        ) from exc

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
            "text1": text1,
            "text2": text2,
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
    # 官方配置：排除neutral，只保留 entailment(1) 和 contradiction(0) 二分类
    label_scores = {
        "entailment": 1.0,
        "contradiction": 0.0,
        0: 1.0,  # entailment numeric
        2: 0.0,  # contradiction numeric
        # neutral (1) is excluded
    }
    pairs: List[Dict[str, object]] = []
    for example in dataset:
        premise = example.get("premise")
        hypothesis = example.get("hypothesis")
        label = example.get("label")
        if premise is None or hypothesis is None or label is None:
            continue
        # 官方做法：跳过 neutral 样本
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
        # 尝试从多个可能的字段名中获取数据（兼容不同数据源）
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
