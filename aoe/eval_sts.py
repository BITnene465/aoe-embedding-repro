"""MTEB-based STS evaluation entry point for AoE checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pkgutil
from importlib import import_module
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from mteb import MTEB

from aoe.model import SentenceEncoder


DEFAULT_TASKS = [
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
    "SICK-R",
]

TASK_IMPORTS = {
    "sts12": ("mteb.tasks.STS.STS12", "STS12"),
    "sts13": ("mteb.tasks.STS.STS13", "STS13"),
    "sts14": ("mteb.tasks.STS.STS14", "STS14"),
    "sts15": ("mteb.tasks.STS.STS15", "STS15"),
    "sts16": ("mteb.tasks.STS.STS16", "STS16"),
    "stsbenchmark": ("mteb.tasks.STS.STSBenchmark", "STSBenchmark"),
    "sick-r": ("mteb.tasks.STS.SICKR", "SICKR"),
    "sickr": ("mteb.tasks.STS.SICKR", "SICKR"),
}
TASK_ALIASES = {
    "stsb": "stsbenchmark",
    "sts-b": "stsbenchmark",
    "sickr": "sickr",
    "sick-r": "sickr",
}


def load_encoder_from_ckpt(ckpt: str, model_cache: str | None = None) -> SentenceEncoder:
    """Instantiate a SentenceEncoder from a full-object checkpoint."""

    ckpt_path = os.path.join(ckpt, "encoder.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Expected checkpoint at {ckpt_path}")

    encoder = torch.load(ckpt_path, map_location="cpu")
    encoder.eval()
    return encoder


def _ensure_data_cache(root_dir: str) -> None:
    """Route all HuggingFace/MTEB caches into the provided repository-local folder."""

    base = Path(root_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    datasets_cache = base / "hf_datasets"
    hub_cache = base / "hf_hub"
    models_cache = base / "hf_models"
    home_cache = base / "hf_home"

    for path in (datasets_cache, hub_cache, models_cache, home_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(home_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(models_cache))
    
    # Enforce offline mode to prevent accidental downloads during evaluation
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _parse_list_argument(raw: str | None, fallback: Iterable[str]) -> list[str]:
    if raw is None or not raw.strip():
        return list(fallback)
    entries = [item.strip() for item in raw.split(",") if item.strip()]
    return entries or list(fallback)


def _normalize_task_name(name: str) -> str:
    return name.replace("-", "").replace("_", "").lower()


def _discover_mteb_task_classes() -> dict[str, type]:
    """Search the installed mteb package for task classes keyed by normalized name."""
    try:
        import mteb.tasks as tasks_pkg
    except Exception as exc:  # pragma: no cover - defensive guard for missing installs
        raise ImportError("mteb is not installed or cannot be imported") from exc

    discovered: dict[str, type] = {}
    prefix = tasks_pkg.__name__ + "."
    for module_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix):
        try:
            module = import_module(module_info.name)
        except Exception:
            # Some optional-task modules may fail to import because of missing deps.
            continue
        for attr in dir(module):
            obj = getattr(module, attr)
            if not isinstance(obj, type):
                continue
            meta = getattr(obj, "metadata", None)
            if meta is None:
                continue
            meta_name = getattr(meta, "name", None)
            if not isinstance(meta_name, str):
                continue
            discovered[_normalize_task_name(meta_name)] = obj
    return discovered


def _import_task_class(module_name: str, class_name: str):
    try:
        module = import_module(module_name)
        return getattr(module, class_name)
    except Exception:
        return None


def _instantiate_mteb_tasks(task_names: Sequence[str]):
    discovered: dict[str, type] | None = None
    available_keys: set[str] = set()
    tasks = []
    for name in task_names:
        key = _normalize_task_name(name)
        if not key:
            continue
        key = TASK_ALIASES.get(key, key)

        # Prefer legacy hardcoded paths for older mteb versions.
        task_cls = None
        target = TASK_IMPORTS.get(key)
        if target is not None:
            module_name, class_name = target
            task_cls = _import_task_class(module_name, class_name)

        # Fallback: dynamically discover tasks shipped with the installed mteb.
        if task_cls is None:
            if discovered is None:
                discovered = _discover_mteb_task_classes()
                available_keys = set(discovered.keys())
            task_cls = discovered.get(key)

        if task_cls is None:
            known = sorted(available_keys or set(TASK_IMPORTS.keys()))
            raise ValueError(f"Task '{name}' is not supported by the installed mteb. Known: {known}")

        tasks.append(task_cls())

    if not tasks:
        raise ValueError("No valid MTEB tasks specified")
    return tasks


def _materialize_sentences(sentences: Iterable[str] | Sequence[str]) -> list[str]:
    """Allow non-subscriptable iterables (e.g., DataLoader) and coerce to List[str]."""

    def _collect(obj) -> list[str]:
        if obj is None:
            return []
        if isinstance(obj, str):
            return [obj]
        # numpy scalar strings (e.g., np.str_) are not instances of str
        if hasattr(obj, "item") and not isinstance(obj, (list, tuple, dict)):
            try:
                return [str(obj.item())]
            except Exception:
                pass
        if isinstance(obj, dict):
            # Prefer common text fields; fall back to the first value.
            for key in ("text", "texts", "sentence", "sentences", "sentence1"):
                if key in obj:
                    return _collect(obj[key])
            if "sentence2" in obj:
                return _collect(obj["sentence2"])
            if obj:
                return _collect(next(iter(obj.values())))
            return []
        if isinstance(obj, (list, tuple, set)):
            out: list[str] = []
            for entry in obj:
                out.extend(_collect(entry))
            return out
        try:
            # torch/numpy arrays, generators, DataLoader, etc.
            return _collect(list(obj))
        except Exception:
            return [str(obj)]

    collected = _collect(sentences)
    return [str(x) for x in collected]


def _to_jsonable(obj):
    """Convert MTEB TaskResult objects (and nested structures) into JSON-friendly data."""

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            pass
    if dataclasses.is_dataclass(obj):
        return _to_jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "to_dict"):
        try:
            return _to_jsonable(obj.to_dict())
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    return str(obj)


class AoEMTEBModel:
    """Adapter that exposes AoE encoders through the interface expected by MTEB."""

    def __init__(
        self,
        encoder: SentenceEncoder,
        device: torch.device,
        max_length: int,
        batch_size: int,
        normalize: bool,
    ) -> None:
        self.encoder = encoder
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize

    @torch.no_grad()
    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int | None = None,
        show_progress_bar: bool = False,
        **_: object,
    ) -> np.ndarray:
        sentences = _materialize_sentences(sentences)
        if not sentences:
            return np.zeros((0, 1), dtype=np.float32)

        bs = batch_size or self.batch_size
        chunks = []
        iterator = range(0, len(sentences), bs)
        for start in iterator:
            batch = sentences[start : start + bs]
            encoded = self.encoder.encode(
                batch,
                device=self.device,
                max_length=self.max_length,
            )
            if isinstance(encoded, tuple):
                encoded = torch.cat(encoded, dim=-1)
            if self.normalize:
                encoded = F.normalize(encoded, p=2, dim=-1)
            chunks.append(encoded.detach().cpu().numpy())

        return np.concatenate(chunks, axis=0)


def main() -> None:
    """CLI entry point for MTEB STS evaluation using AoE checkpoints."""

    parser = argparse.ArgumentParser(description="Run MTEB STS tasks with an AoE checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint directory")
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated list of MTEB task names (default: STS suite)",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--data_cache",
        default="data",
        help="Directory where HuggingFace datasets/models will be cached",
    )
    parser.add_argument("--model_cache", default="models")
    parser.add_argument(
        "--results_dir",
        default="output/mteb",
        help="Where to store detailed MTEB outputs",
    )
    parser.add_argument(
        "--eval_splits",
        default="test",
        help="Comma-separated list of splits to evaluate (default: test)",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Identifier for this checkpoint in MTEB outputs (default: ckpt folder name)",
    )
    parser.add_argument(
        "--no_l2_norm",
        action="store_true",
        help="Disable L2 normalization of embeddings before similarity (defaults to on)",
    )
    args = parser.parse_args()

    _ensure_data_cache(args.data_cache)

    encoder = load_encoder_from_ckpt(args.ckpt, model_cache=args.model_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    tasks = _parse_list_argument(args.tasks, DEFAULT_TASKS)
    splits = _parse_list_argument(args.eval_splits, ["test"])

    model_name = args.model_name or Path(args.ckpt).resolve().name

    adapter = AoEMTEBModel(
        encoder=encoder,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        normalize=not args.no_l2_norm,
    )
    task_objects = _instantiate_mteb_tasks(tasks)
    suite = MTEB(tasks=task_objects)
    results_dir = Path(args.results_dir) / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    results = suite.run(
        adapter,
        eval_splits=splits,
        output_folder=str(results_dir),
        model_name=model_name,
        overwrite_results=True,  # ensure every task reruns for this checkpoint
    )
    results_jsonable = _to_jsonable(results)

    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(results_jsonable, fp, indent=2, ensure_ascii=False)

    print(f"MTEB results saved to {summary_path}")
    print(json.dumps(results_jsonable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
