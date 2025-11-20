"""Data loading and preprocessing utilities for AoE experiments."""

from typing import Dict

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


def load_nli_dataset(split: str = "train") -> Dataset:
	"""Load SNLI and MultiNLI, clean invalid labels, and merge them into one dataset."""

	snli = load_dataset("snli", split=split).filter(_filter_valid_labels)

	mnli_split = _resolve_mnli_split(split)
	mnli = load_dataset("multi_nli", split=mnli_split).filter(_filter_valid_labels)

	required_cols = {"premise", "hypothesis", "label"}
	if not required_cols.issubset(snli.column_names):
		raise ValueError("SNLI split is missing required fields")
	if not required_cols.issubset(mnli.column_names):
		raise ValueError("MultiNLI split is missing required fields")

	combined = concatenate_datasets([snli, mnli])
	return combined


def load_stsb_splits() -> Dict[str, Dataset]:
    """Return STS-B splits with fields (sentence1, sentence2, score)."""

    ds_dict = load_dataset("glue", "stsb")
    splits: Dict[str, Dataset] = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in ds_dict:
            continue
        split_ds = ds_dict[split_name]
        if "label" in split_ds.column_names and "score" not in split_ds.column_names:
            split_ds = split_ds.rename_column("label", "score")
        splits[split_name] = split_ds
    return splits


def load_sickr_split() -> Dataset:
    """Load the SICK-R dataset (validation split) with normalized field names."""

    try:
        ds = load_dataset("sick", split="validation")
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


def load_gis_splits() -> Dict[str, Dataset]:
    """Load the GitHub Issue Similarity dataset and expose sentence fields plus scores."""

    ds_dict = load_dataset("WhereIsAI/github-issue-similarity")

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
