"""Loss functions implementing the AoE angle-based objectives."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
	embeddings: torch.Tensor,
	labels: torch.Tensor,
	temperature: float = 0.05,
) -> torch.Tensor:
	"""Compute SimCSE-style supervised contrastive loss.

	Args:
		embeddings: Real-valued sentence embeddings of shape [batch, dim].
		labels: Integer labels of shape [batch]; equal labels form positives.
		temperature: Softmax temperature applied to cosine similarities.

	Returns:
		Scalar tensor containing the supervised contrastive loss.
	"""

	if embeddings.dim() != 2:
		raise ValueError("embeddings must be 2D (batch, dim)")

	if labels.dim() != 1 or labels.size(0) != embeddings.size(0):
		raise ValueError("labels must be 1D and aligned with embeddings")

	device = embeddings.device
	batch_size = embeddings.size(0)
	if batch_size <= 1:
		return embeddings.new_tensor(0.0)

	normalized = F.normalize(embeddings, p=2, dim=1)
	similarity = torch.matmul(normalized, normalized.T)
	logits = similarity / temperature

	eye = torch.eye(batch_size, device=device, dtype=torch.bool)
	all_mask = ~eye
	labels_equal = labels.view(-1, 1) == labels.view(1, -1)
	positive_mask = labels_equal & all_mask

	exp_logits = torch.exp(logits) * all_mask
	denom = exp_logits.sum(dim=1, keepdim=True) + 1e-12
	log_prob = logits - torch.log(denom)

	positive_counts = positive_mask.sum(dim=1)
	loss_per_sample = -(log_prob * positive_mask.float()).sum(dim=1) / (
		positive_counts.float() + 1e-12
	)

	valid_mask = positive_counts > 0
	if valid_mask.sum() == 0:
		return embeddings.new_tensor(0.0)

	return loss_per_sample[valid_mask].mean()


def _pairwise_angle_delta(z_re: torch.Tensor, z_im: torch.Tensor) -> torch.Tensor:
	"""Return absolute angle difference matrix for all embedding pairs."""

	# Promote to float32 for numerical stability before any matmuls.
	z_re = z_re.float()
	z_im = z_im.float()
	# Complex dot products across all pairs.
	dot_re = torch.matmul(z_re, z_re.T) + torch.matmul(z_im, z_im.T)
	dot_im = torch.matmul(z_re, z_im.T) - torch.matmul(z_im, z_re.T)
	norm = torch.sqrt((z_re.pow(2) + z_im.pow(2)).sum(dim=1).clamp_min(1e-12))
	denom = torch.outer(norm, norm).clamp_min(1e-12)
	cos_theta = (dot_re / denom).clamp(-1.0, 1.0)
	sin_theta = dot_im / denom
	return torch.atan2(sin_theta, cos_theta).abs()


def angle_loss(
	z_re: torch.Tensor,
	z_im: torch.Tensor,
	labels: torch.Tensor,
	temperature: float = 0.05,
) -> torch.Tensor:
	"""Compute AoE angle-ranking loss using all positive/negative pairs."""

	if z_re.shape != z_im.shape:
		raise ValueError("z_re and z_im must share the same shape")

	batch_size = z_re.size(0)
	if labels.size(0) != batch_size:
		raise ValueError("labels must align with embeddings")

	if batch_size <= 1:
		return z_re.new_tensor(0.0)

	device = z_re.device
	same_label = labels.view(-1, 1) == labels.view(1, -1)
	eye = torch.eye(batch_size, device=device, dtype=torch.bool)
	pos_mask = same_label & ~eye
	neg_mask = ~same_label

	if not pos_mask.any() or not neg_mask.any():
		return z_re.new_tensor(0.0)

	delta = _pairwise_angle_delta(z_re, z_im)
	theta_pos = delta.unsqueeze(2)
	theta_neg = delta.unsqueeze(1)
	valid = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
	if not valid.any():
		return z_re.new_tensor(0.0)

	diff = (theta_pos - theta_neg) / temperature
	loss_matrix = F.softplus(diff)
	loss = (loss_matrix * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
	return loss


def aoe_total_loss(
	z_re: torch.Tensor,
	z_im: torch.Tensor,
	labels: torch.Tensor,
	tau_angle: float = 0.05,
	tau_cl: float = 0.05,
	w_angle: float = 1.0,
	w_cl: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
	"""Combine angle ranking and supervised contrastive losses."""

	z = torch.cat([z_re, z_im], dim=-1)
	cl_loss = supervised_contrastive_loss(z, labels, tau_cl)
	ang_loss = angle_loss(z_re, z_im, labels, tau_angle)
	total = w_angle * ang_loss + w_cl * cl_loss

	stats = {
		"angle_loss": float(ang_loss.item()),
		"contrastive_loss": float(cl_loss.item()),
		"total_loss": float(total.item()),
	}
	return total, stats
