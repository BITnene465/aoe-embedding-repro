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


def _angle_distance(
	z_re_a: torch.Tensor,
	z_im_a: torch.Tensor,
	z_re_b: torch.Tensor,
	z_im_b: torch.Tensor,
) -> torch.Tensor:
	"""Return absolute angle difference between two complex vectors."""

	dot_re = (z_re_a * z_re_b + z_im_a * z_im_b).sum(dim=1)
	dot_im = (z_re_a * z_im_b - z_im_a * z_re_b).sum(dim=1)
	norm_a = torch.sqrt((z_re_a.pow(2) + z_im_a.pow(2)).sum(dim=1) + 1e-12)
	norm_b = torch.sqrt((z_re_b.pow(2) + z_im_b.pow(2)).sum(dim=1) + 1e-12)
	denom = norm_a * norm_b + 1e-12
	cos_theta = (dot_re / denom).clamp(-1.0, 1.0)
	sin_theta = dot_im / denom
	delta = torch.atan2(sin_theta, cos_theta)
	return delta.abs()


def angle_loss(
	z_re: torch.Tensor,
	z_im: torch.Tensor,
	labels: torch.Tensor,
	temperature: float = 0.05,
) -> torch.Tensor:
	"""Compute simplified AoE angle-ranking loss for complex embeddings."""

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

	anchors, positives, negatives = [], [], []
	for idx in range(batch_size):
		pos_candidates = torch.nonzero(pos_mask[idx], as_tuple=False).view(-1)
		neg_candidates = torch.nonzero(neg_mask[idx], as_tuple=False).view(-1)
		if pos_candidates.numel() == 0 or neg_candidates.numel() == 0:
			continue
		anchors.append(idx)
		positives.append(pos_candidates[0].item())
		negatives.append(neg_candidates[0].item())

	if not anchors:
		return z_re.new_tensor(0.0)

	anchor_idx = torch.tensor(anchors, device=device, dtype=torch.long)
	pos_idx = torch.tensor(positives, device=device, dtype=torch.long)
	neg_idx = torch.tensor(negatives, device=device, dtype=torch.long)

	theta_pos = _angle_distance(
		z_re[anchor_idx], z_im[anchor_idx], z_re[pos_idx], z_im[pos_idx]
	)
	theta_neg = _angle_distance(
		z_re[anchor_idx], z_im[anchor_idx], z_re[neg_idx], z_im[neg_idx]
	)

	loss = F.softplus((theta_pos - theta_neg) / temperature).mean()
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
