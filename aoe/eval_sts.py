"""MTEB-based STS evaluation entry point for AoE checkpoints."""

from __future__ import annotations

import argparse
import json
import pkgutil
from importlib import import_module
from pathlib import Path
from typing import Iterable, Sequence

import torch
from mteb import MTEB

from aoe.eval_utils import (
    AoEMTEBModel,
    ensure_data_cache,
    load_encoder_from_ckpt,
    to_jsonable,
)


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
    "sickr": ("mteb.tasks.STS.SICKR", "SICKR"),
    "sick-r": ("mteb.tasks.STS.SICKR", "SICKR"),
}
TASK_ALIASES = {
    "stsb": "stsbenchmark",
    "sts-b": "stsbenchmark",
    "sickr": "sickr",
    "sick-r": "sickr",
}


def _parse_list_argument(raw: str | None, fallback: Iterable[str]) -> list[str]:
    if raw is None or not raw.strip():
        return list(fallback)
    entries = [item.strip() for item in raw.split(",") if item.strip()]
    if "all" in entries:
        return list(fallback)
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

    ensure_data_cache(args.data_cache)

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
    results_jsonable = to_jsonable(results)

    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(results_jsonable, fp, indent=2, ensure_ascii=False)

    print(f"MTEB results saved to {summary_path}")
    print(json.dumps(results_jsonable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
