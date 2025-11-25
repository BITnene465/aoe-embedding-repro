"""Script to download datasets for AoE reproduction."""

import argparse
import os
from typing import List
from datasets import load_dataset

def setup_hf_env(cache_dir: str):
    """Set up robust environment variables for HuggingFace."""
    # Define all relevant cache paths
    hf_home = os.path.join(cache_dir, "hf_home")
    datasets_cache = os.path.join(cache_dir, "hf_datasets")
    modules_cache = os.path.join(cache_dir, "hf_modules")
    metrics_cache = os.path.join(cache_dir, "hf_metrics")
    hub_cache = os.path.join(cache_dir, "hf_hub")

    # Set environment variables (covering various versions of HF libs)
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = datasets_cache
    os.environ["HF_MODULES_CACHE"] = modules_cache
    os.environ["HF_METRICS_CACHE"] = metrics_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    
    # Legacy/Alternative names
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "hf_models")
    os.environ["XDG_CACHE_HOME"] = os.path.join(cache_dir, "xdg_cache")

    # Create directories
    for path in [hf_home, datasets_cache, modules_cache, metrics_cache, hub_cache, os.environ["TRANSFORMERS_CACHE"]]:
        os.makedirs(path, exist_ok=True)

    print(f"Environment variables set for HF cache at: {cache_dir}")

def download_nli(output_dir: str):
    print("Downloading SNLI...")
    snli = load_dataset("snli")
    snli.save_to_disk(os.path.join(output_dir, "snli"))

    print("Downloading MultiNLI...")
    mnli = load_dataset("multi_nli")
    mnli.save_to_disk(os.path.join(output_dir, "multi_nli"))

def download_stsb(output_dir: str):
    print("Downloading STS-B...")
    stsb = load_dataset("glue", "stsb")
    stsb.save_to_disk(os.path.join(output_dir, "stsb"))

def download_sickr(output_dir: str):
    print("Downloading SICK-R...")
    try:
        sickr = load_dataset("mteb/sickr-sts")
        sickr.save_to_disk(os.path.join(output_dir, "sickr"))
    except Exception as e:
        print(f"Failed to download SICK-R: {e}")

def download_gis(output_dir: str):
    print("Downloading GIS...")
    gis = load_dataset("WhereIsAI/github-issue-similarity")
    gis.save_to_disk(os.path.join(output_dir, "gis"))

def download_mteb_sts(output_dir: str):
    """Download MTEB STS tasks (STS12-16) to HF cache only."""
    # These are used by MTEB evaluation, not directly by aoe/data.py for training.
    # We just need to load them so they get cached in HF_DATASETS_CACHE.
    tasks = ["mteb/sts12-sts", "mteb/sts13-sts", "mteb/sts14-sts", "mteb/sts15-sts", "mteb/sts16-sts"]
    print(f"Downloading MTEB STS tasks ({', '.join(tasks)}) to cache...")
    for task in tasks:
        try:
            print(f"  - {task}")
            load_dataset(task)
        except Exception as e:
            print(f"  - Failed to download {task}: {e}")

DOWNLOAD_FUNCS = {
    "nli": download_nli,
    "stsb": download_stsb,
    "sickr": download_sickr,
    "gis": download_gis,
    "mteb_sts": download_mteb_sts,
}

def download_data(output_dir: str, cache_dir: str, datasets: List[str]):
    """Download specified datasets."""
    setup_hf_env(cache_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving processed data to: {output_dir}")

    to_download = set()
    if "all" in datasets:
        to_download = set(DOWNLOAD_FUNCS.keys())
    else:
        for d in datasets:
            if d in DOWNLOAD_FUNCS:
                to_download.add(d)
            else:
                print(f"Warning: Unknown dataset '{d}', skipping. Available: {list(DOWNLOAD_FUNCS.keys())}")

    for name in to_download:
        print(f"--- Processing {name} ---")
        DOWNLOAD_FUNCS[name](output_dir)

    print(f"Finished downloading: {', '.join(to_download)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for AoE")
    parser.add_argument("--output_dir", default="data", help="Directory to save processed datasets (for training)")
    parser.add_argument("--cache_dir", default="data", help="Directory for HF cache (for MTEB evaluation)")
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["all"], 
        choices=list(DOWNLOAD_FUNCS.keys()) + ["all"],
        help="List of datasets to download (default: all)"
    )
    args = parser.parse_args()

    download_data(args.output_dir, args.cache_dir, args.datasets)
