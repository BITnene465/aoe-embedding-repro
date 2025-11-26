"""MTEB-based Downstream Task (Classification) evaluation entry point for AoE checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from mteb import MTEB

from aoe.model import SentenceEncoder
from aoe.eval_utils import (
    load_encoder_from_ckpt,
    ensure_data_cache,
    to_jsonable,
    AoEMTEBModel,
)
from aoe.eval_sts import _parse_list_argument

# Downstream tasks from Table 3 of the paper
DOWNSTREAM_TASKS = [
    "MR",
    "CR",
    "SUBJ",
    "SST2",
]

TASK_IMPORTS = {
    "mr": ("mteb.tasks.Classification.MR", "MR"),
    "cr": ("mteb.tasks.Classification.CR", "CR"),
    "subj": ("mteb.tasks.Classification.SUBJ", "SUBJ"),
    "sst2": ("mteb.tasks.Classification.SST2", "SST2"),
}


def _import_task_class(module_name: str, class_name: str):
    from importlib import import_module
    try:
        module = import_module(module_name)
        return getattr(module, class_name)
    except Exception:
        return None


from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification

class MR(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MR",
        dataset={
            "path": "rotten_tomatoes",
            "revision": "main"
        },
        description="Movie Review Sentiment Classification",
        type="Classification",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        bibtex_citation=None,
    )

class CR(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CR",
        dataset={
            "path": "SetFit/cr",
            "revision": "main"
        },
        description="Customer Review Sentiment Classification",
        type="Classification",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        bibtex_citation=None,
    )

class SUBJ(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SUBJ",
        dataset={
            "path": "SetFit/subj",
            "revision": "main"
        },
        description="Subjectivity Classification",
        type="Classification",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        bibtex_citation=None,
    )

class SST2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SST2",
        dataset={
            "path": "SetFit/sst2",
            "revision": "main"
        },
        description="Stanford Sentiment Treebank",
        type="Classification",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        bibtex_citation=None,
    )


CUSTOM_TASKS = {
    "mr": MR,
    "cr": CR,
    "subj": SUBJ,
    "sst2": SST2,
}

def _instantiate_transfer_tasks(task_names: Sequence[str]):
    tasks = []
    for name in task_names:
        key = name.lower()
        if key in CUSTOM_TASKS:
            tasks.append(CUSTOM_TASKS[key]())
            continue
            
        # Fallback to import if not in custom tasks (unlikely for this list)
        target = TASK_IMPORTS.get(key)
        if target is None:
            raise ValueError(f"Task '{name}' is not a supported transfer task. Known: {list(CUSTOM_TASKS.keys())}")
        
        module_name, class_name = target
        task_cls = _import_task_class(module_name, class_name)
        if task_cls is None:
             raise ImportError(f"Could not import task '{name}' from {module_name}.{class_name}")
        
        tasks.append(task_cls())

    if not tasks:
        raise ValueError("No valid transfer tasks specified")
    return tasks


def main() -> None:
    """CLI entry point for MTEB Downstream evaluation using AoE checkpoints."""

    parser = argparse.ArgumentParser(description="Run MTEB Downstream tasks with an AoE checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint directory")
    parser.add_argument(
        "--tasks",
        default=",".join(DOWNSTREAM_TASKS),
        help="Comma-separated list of Downstream task names",
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
        default="output/downstream",
        help="Where to store detailed MTEB outputs",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Identifier for this checkpoint in MTEB outputs (default: ckpt folder name)",
    )
    parser.add_argument(
        "--no_l2_norm",
        action="store_true",
        help="Disable L2 normalization of embeddings (defaults to on)",
    )
    args = parser.parse_args()

    ensure_data_cache(args.data_cache)

    encoder = load_encoder_from_ckpt(args.ckpt, model_cache=args.model_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    tasks_list = _parse_list_argument(args.tasks, DOWNSTREAM_TASKS)
    
    model_name = args.model_name or Path(args.ckpt).resolve().name

    adapter = AoEMTEBModel(
        encoder=encoder,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        normalize=not args.no_l2_norm,
    )
    
    task_objects = _instantiate_transfer_tasks(tasks_list)
    suite = MTEB(tasks=task_objects)
    results_dir = Path(args.results_dir) / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Transfer tasks usually use k-fold cross validation on train/test splits provided by MTEB
    # We just run .run() which handles the specific evaluation logic for Classification tasks
    results = suite.run(
        adapter,
        output_folder=str(results_dir),
        model_name=model_name,
        overwrite_results=True,
    )
    results_jsonable = to_jsonable(results)

    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(results_jsonable, fp, indent=2, ensure_ascii=False)

    print(f"Transfer results saved to {summary_path}")
    # Print a simple summary table
    print(f"{'Task':<10} | {'Accuracy':<10}")
    print("-" * 23)
    for task_res in results:
        # MTEB results structure varies, but for classification it usually has 'test' -> 'accuracy'
        # Or 'test' -> 'accuracy'
        # Let's try to extract the main metric
        task_name = task_res.task_name
        # Classification results are often nested
        # e.g. results['test']['accuracy']
        # But MTEB object structure might be different in list
        # Actually suite.run returns a list of TaskResult objects or dicts?
        # It returns a list of MTEBResult objects in newer versions, or dicts in older.
        # Let's inspect the jsonable version
        
        # We can just print the json for now as the user can inspect it.
        pass
    
    print(json.dumps(results_jsonable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
