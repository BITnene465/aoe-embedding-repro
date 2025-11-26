"""MTEB-based GIS evaluation entry point for AoE checkpoints."""

from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path
from mteb import MTEB, TaskMetadata
from mteb.abstasks import AbsTaskSTS

from aoe.data import load_gis_splits
from aoe.eval_utils import (
    AoEMTEBModel,
    ensure_data_cache,
    load_encoder_from_ckpt,
    to_jsonable,
)

class GISTask(AbsTaskSTS):
    metadata = TaskMetadata(
        name="GIS",
        description="GitHub Issue Similarity",
        reference=None,
        type="STS",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="cosine_spearman",
        dataset={
            "path": "WhereIsAI/github-issue-similarity",
            "revision": "main"
        },
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        bibtex_citation=None,
    )
    
    # Allow configuring cache_dir externally
    data_cache_dir = "data"

    def load_data(self, **kwargs):
        self.dataset = load_gis_splits(cache_dir=self.data_cache_dir)
        # Rename columns to match MTEB expectations
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].rename_columns(
                {"text1": "sentence1", "text2": "sentence2"}
            )


def main() -> None:
    """CLI entry point for GIS evaluation using AoE checkpoints."""

    parser = argparse.ArgumentParser(description="Run GIS evaluation with an AoE checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint directory")
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

    # Configure GIS task cache
    GISTask.data_cache_dir = args.data_cache

    model_name = args.model_name or Path(args.ckpt).resolve().name

    adapter = AoEMTEBModel(
        encoder=encoder,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        normalize=not args.no_l2_norm,
    )
    
    # Instantiate GIS task
    task = GISTask()
    suite = MTEB(tasks=[task])
    
    results_dir = Path(args.results_dir) / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    results = suite.run(
        adapter,
        eval_splits=["test"],
        output_folder=str(results_dir),
        model_name=model_name,
        overwrite_results=True,
    )
    results_jsonable = to_jsonable(results)

    summary_path = results_dir / "summary_gis.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(results_jsonable, fp, indent=2, ensure_ascii=False)

    print(f"GIS results saved to {summary_path}")
    print(json.dumps(results_jsonable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
