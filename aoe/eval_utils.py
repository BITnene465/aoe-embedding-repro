"""Shared utilities for AoE evaluation scripts."""

import dataclasses
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from aoe.model import SentenceEncoder
from mteb.models.model_meta import ModelMeta

def load_encoder_from_ckpt(ckpt: str, model_cache: str | None = None) -> SentenceEncoder:
    """Instantiate a SentenceEncoder from a full-object checkpoint."""

    ckpt_path = os.path.join(ckpt, "encoder.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Expected checkpoint at {ckpt_path}")

    encoder = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder.eval()
    return encoder


def ensure_data_cache(root_dir: str) -> None:
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


def materialize_sentences(sentences: Iterable[str] | Sequence[str]) -> list[str]:
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


def to_jsonable(obj):
    """Convert MTEB TaskResult objects (and nested structures) into JSON-friendly data."""

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            pass
    if dataclasses.is_dataclass(obj):
        return to_jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "to_dict"):
        try:
            return to_jsonable(obj.to_dict())
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
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
        sentences = materialize_sentences(sentences)
        if not sentences:
            return np.zeros((0, 1), dtype=np.float32)

        bs = batch_size or self.batch_size
        chunks = []
        iterator = range(0, len(sentences), bs)
        for start in iterator:
            batch = sentences[start : start + bs]
            # output_mlp=False is default, consistent with inference
            encoded = self.encoder.encode(
                batch,
                device=self.device,
                max_length=self.max_length,
            )
            if self.normalize:
                encoded = F.normalize(encoded, p=2, dim=-1)
            chunks.append(encoded.detach().cpu().numpy())

        return np.concatenate(chunks, axis=0)

    def similarity(self, embeddings1, embeddings2):
        """Compute cosine similarity between two collections of embeddings."""
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        
        embeddings1 = embeddings1.to(self.device)
        embeddings2 = embeddings2.to(self.device)
        
        # Normalize if not already (though encode usually does it if configured)
        # But to be safe and strictly follow cosine similarity definition:
        embeddings1 = F.normalize(embeddings1, p=2, dim=-1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=-1)
        
        return (embeddings1 @ embeddings2.T)

    def similarity_pairwise(self, embeddings1, embeddings2):
        """Compute pairwise cosine similarity."""
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
            
        embeddings1 = embeddings1.to(self.device)
        embeddings2 = embeddings2.to(self.device)
        
        return F.cosine_similarity(embeddings1, embeddings2, dim=-1)

    @property
    def mteb_model_meta(self) -> ModelMeta:
        """Metadata of the model."""
        return ModelMeta(
            name="AoE/AoE-Model",
            revision="main",
            release_date=None,
            languages=["eng-Latn"],
            loader=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=self.max_length,
            embed_dim=768,  # Assuming BERT base
            license="mit",
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )
