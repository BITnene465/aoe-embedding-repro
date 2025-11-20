"""Model definitions for reproducing AoE sentence encoders."""

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
	) -> None:
		"""Initialize tokenizer, encoder, and pooling strategy."""

		super().__init__()
		if pooling not in {"cls", "mean"}:
			raise ValueError("pooling must be either 'cls' or 'mean'")

		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModel.from_pretrained(model_name)
		self.complex_mode = complex_mode
		self.pooling = pooling

		for param in self.model.parameters():
			param.requires_grad = True

	def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
		"""Aggregate token representations using the configured pooling strategy."""

		if self.pooling == "cls":
			return hidden_states[:, 0]

		mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
		masked_hidden = hidden_states * mask
		denom = mask.sum(dim=1).clamp(min=1e-9)
		return masked_hidden.sum(dim=1) / denom

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

