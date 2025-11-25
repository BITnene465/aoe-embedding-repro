"""Model definitions for reproducing AoE sentence encoders."""

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SentenceEncoder(nn.Module):
	"""Lightweight wrapper around a BERT encoder with AoE-style outputs."""

	def __init__(
		self,
		model_name: str = "bert-base-uncased",
		complex_mode: bool = False,
		pooling: str = "cls",
		cache_dir: Optional[str] = "models",
		prompt: Optional[str] = None,
	) -> None:
		"""Initialize tokenizer, encoder, and pooling strategy.
		
		Args:
			model_name: HuggingFace model name.
			complex_mode: Whether to output complex embeddings (real+imag).
			pooling: Pooling strategy ('cls', 'mean', 'cls_avg', 'max').
			cache_dir: Directory where models are stored locally.
			prompt: Optional prompt template to use during inference. 
					Note: This class doesn't automatically apply the prompt in encode(), 
					but stores it for reference/reproducibility.
		"""

		super().__init__()
		if pooling not in {"cls", "mean", "cls_avg", "max"}:
			raise ValueError("pooling must be one of: cls, mean, cls_avg, max")

		# Ensure we look for the model in the local cache directory if it's not a full path
		model_path = model_name
		if cache_dir and not os.path.isabs(model_name) and not os.path.exists(model_name):
			possible_path = os.path.join(cache_dir, model_name)
			if os.path.exists(possible_path):
				model_path = possible_path
		
		# Enforce local loading
		try:
			self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
			self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
		except OSError as e:
			raise OSError(
				f"Could not load model from '{model_path}'. "
				f"Please ensure it is downloaded to '{cache_dir}' using scripts/download/download_model.py"
			) from e

		self.complex_mode = complex_mode
		self.pooling = pooling
		self.cache_dir = cache_dir
		self.model_name = model_name
		self.prompt = prompt

		for param in self.model.parameters():
			param.requires_grad = True

	def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
		"""Aggregate token representations using the configured pooling strategy."""

		if self.pooling == "cls":
			return hidden_states[:, 0]

		if self.pooling == "max":
			mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
			# Fill padding with very small value before max
			masked_hidden = hidden_states.masked_fill(~mask.bool(), -1e9)
			return torch.max(masked_hidden, dim=1)[0]

		# Mean pooling logic
		mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
		masked_hidden = hidden_states * mask
		denom = mask.sum(dim=1).clamp(min=1e-9)
		mean_pooled = masked_hidden.sum(dim=1) / denom

		if self.pooling == "mean":
			return mean_pooled

		if self.pooling == "cls_avg":
			return (hidden_states[:, 0] + mean_pooled) / 2.0

		raise ValueError(f"Unsupported pooling strategy: {self.pooling}")

	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: torch.Tensor,
		token_type_ids: Optional[torch.Tensor] = None,
	) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
		"""Compute embeddings from pre-tokenized inputs."""
		outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
		hidden_states = outputs.last_hidden_state
		sentence_repr = self._pool(hidden_states, attention_mask)

		if not self.complex_mode:
			return sentence_repr

		hidden_dim = sentence_repr.size(-1)
		if hidden_dim % 2 != 0:
			raise ValueError("Hidden dimension must be even for complex mode")

		dim_half = hidden_dim // 2
		z_re = sentence_repr[:, :dim_half]
		z_im = sentence_repr[:, dim_half:]
		return z_re, z_im

	def encode(
		self,
		texts: List[str],
		device: Optional[torch.device] = None,
		max_length: int = 512,
	) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
		"""Encode text inputs into real or complex sentence embeddings.

		Returns real-valued vectors when ``complex_mode`` is ``False`` and a tuple
		containing real and imaginary parts otherwise.
		"""

		if not texts:
			raise ValueError("texts must contain at least one string")

		encodings = self.tokenizer(
			texts,
			padding=True,
			truncation=True,
			max_length=max_length,
			return_tensors="pt",
		)

		model_device = next(self.model.parameters()).device
		target_device = device or model_device
		encodings = {k: v.to(target_device) for k, v in encodings.items()}

		outputs = self.model(**encodings)
		hidden_states = outputs.last_hidden_state
		sentence_repr = self._pool(hidden_states, encodings["attention_mask"])

		if not self.complex_mode:
			return sentence_repr

		hidden_dim = sentence_repr.size(-1)
		if hidden_dim % 2 != 0:
			raise ValueError("Hidden dimension must be even for complex mode")

		dim_half = hidden_dim // 2
		z_re = sentence_repr[:, :dim_half]
		z_im = sentence_repr[:, dim_half:]
		return z_re, z_im

