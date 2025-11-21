"""AoE / AnglE loss implementations matching the ACL 2024 reference."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def angle_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    tau: float = 20.0,
    pooling: str = "sum",
) -> torch.Tensor:
    """Compute the AnglE ranking loss using pairwise ordering."""
    if y_pred.dim() != 2:
        raise ValueError("y_pred must be a 2D tensor [batch, feat]")
    if y_true.dim() not in (1, 2):
        raise ValueError("y_true must be 1D or 2D")
    batch_size, feat_dim = y_pred.shape
    if batch_size % 2 != 0:
        raise ValueError("Batch size must be even (pairs in zigzag order)")
    if feat_dim % 2 != 0:
        raise ValueError("Feature dimension must be even (real + imag)")
    y_true_flat = y_true.view(batch_size)
    if y_true_flat.size(0) != batch_size:
        raise ValueError("y_true must match the batch dimension of y_pred")

    num_pairs = batch_size // 2
    dim = feat_dim // 2

    pair_scores = y_true_flat[0::2].float()
    order = (pair_scores[:, None] < pair_scores[None, :]).float()

    real, imag = y_pred[:, :dim], y_pred[:, dim:]
    a, b = real[0::2], imag[0::2]
    c, d = real[1::2], imag[1::2]

    denom = (c.pow(2) + d.pow(2)).sum(dim=1, keepdim=True).clamp_min(1e-12)
    re = (a * c + b * d) / denom
    im = (b * c - a * d) / denom

    norm_z = (a.pow(2) + b.pow(2)).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    norm_w = (c.pow(2) + d.pow(2)).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    ratio = norm_z / norm_w
    re = re / ratio
    im = im / ratio

    complex_vec = torch.cat([re, im], dim=1)
    if pooling == "sum":
        pooled = complex_vec.sum(dim=1)
    elif pooling == "mean":
        pooled = complex_vec.mean(dim=1)
    else:
        raise ValueError("pooling must be either 'sum' or 'mean'")

    scores = pooled.abs() * tau
    diff = scores[:, None] - scores[None, :]
    masked_diff = diff - (1.0 - order) * 1e12
    flat = masked_diff.reshape(-1)
    flat = torch.cat([flat.new_zeros(1, device=flat.device, dtype=flat.dtype), flat], dim=0)
    return torch.logsumexp(flat, dim=0).to(torch.float32)


def supervised_contrastive_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    scale: float = 20.0,
) -> torch.Tensor:
    """InfoNCE-style supervised contrastive loss with anchors matched to positives."""
    if anchors.shape != positives.shape:
        raise ValueError("anchors and positives must share the same shape")
    if anchors.dim() != 2:
        raise ValueError("inputs must be 2D tensors [batch, dim]")

    batch_size = anchors.size(0)
    if batch_size <= 1:
        return anchors.new_tensor(0.0, dtype=torch.float32)

    anchors_norm = F.normalize(anchors, p=2, dim=1)
    positives_norm = F.normalize(positives, p=2, dim=1)
    logits = anchors_norm @ positives_norm.T * scale
    targets = torch.arange(batch_size, device=anchors.device)
    return F.cross_entropy(logits, targets).to(torch.float32)


def aoe_total_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    angle_tau: float = 20.0,
    cl_scale: float = 20.0,
    w_angle: float = 0.02,
    w_cl: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combine AnglE ranking loss with supervised contrastive loss."""
    angle_term = angle_loss(y_true, y_pred, tau=angle_tau, pooling="sum")

    anchors = y_pred[0::2]
    positives = y_pred[1::2]
    cl_term = supervised_contrastive_loss(anchors, positives, scale=cl_scale)

    total = w_angle * angle_term + w_cl * cl_term
    stats = {
        "angle_loss": float(angle_term.item()),
        "contrastive_loss": float(cl_term.item()),
        "total_loss": float(total.item()),
    }
    return total, stats