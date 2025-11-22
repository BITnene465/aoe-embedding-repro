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
    order = (pair_scores[:, None] > pair_scores[None, :]).float()

    y_pred_re, y_pred_im = y_pred[:, :dim], y_pred[:, dim:]
    a = y_pred_re[0::2]
    b = y_pred_im[0::2]
    c = y_pred_re[1::2]
    d = y_pred_im[1::2]

    # Complex division: (a+bi) / (c+di) = ((ac+bd) + i(bc-ad)) / (c^2+d^2)
    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    # Normalize by magnitude ratio
    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True)**0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True)**0.5
    re /= (dz / dw)
    im /= (dz / dw)

    # Concatenate re and im, then pool
    y_pred_angle = torch.cat((re, im), dim=1)
    if pooling == "sum":
        pooled = torch.sum(y_pred_angle, dim=1)
    elif pooling == "mean":
        pooled = torch.mean(y_pred_angle, dim=1)
    else:
        raise ValueError("pooling must be either 'sum' or 'mean'")

    # Apply absolute value and scale
    scores = torch.abs(pooled) * tau
    diff = scores[:, None] - scores[None, :]
    if order.sum() == 0:
        return y_pred.new_tensor(0.0, dtype=torch.float32)

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