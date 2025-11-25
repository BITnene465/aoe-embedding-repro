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


def categorical_crossentropy_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    from_logits: bool = True,
) -> torch.Tensor:
    """Compute categorical crossentropy."""
    if from_logits:
        return -(F.log_softmax(y_pred, dim=1) * y_true).sum(dim=1)
    return -(torch.log(y_pred, dim=1) * y_true).sum(dim=1)


def make_target_matrix(y_true: torch.Tensor) -> torch.Tensor:
    """Construct the target matrix for in-batch negative loss."""
    device = y_true.device
    idxs = torch.arange(0, y_true.shape[0]).int().to(device)
    y_true = y_true.int()
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]

    # Expand y_true to match the shape requirements
    y_true_row = y_true[None, :]  # shape: [1, batch_size]
    y_true_col = y_true[:, None]  # shape: [batch_size, 1]

    idxs_1 = idxs_1 * y_true_row
    idxs_1 = idxs_1 + (y_true_row == 0).int() * -2

    idxs_2 = idxs_2 * y_true_col
    idxs_2 = idxs_2 + (y_true_col == 0).int() * -1

    y_true = (idxs_1 == idxs_2).float()
    return y_true


def in_batch_negative_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    tau: float = 20.0,
    negative_weights: float = 0.0,
) -> torch.Tensor:
    """Compute in-batch negative loss (contrastive loss) matching official AnglE."""
    device = y_true.device
    
    # y_true is zigzag [batch_size], e.g. [1, 1, 0, 0, ...]
    # We need to reshape/process it to match the matrix construction logic
    # The official code expects y_true to be the labels for the batch
    
    neg_mask = make_target_matrix(y_true == 0)
    y_true_matrix = make_target_matrix(y_true)

    # compute similarity
    y_pred = F.normalize(y_pred, dim=1, p=2)
    similarities = y_pred @ y_pred.T  # dot product
    similarities = similarities - torch.eye(y_pred.shape[0]).to(device) * 1e12
    similarities = similarities * tau

    if negative_weights > 0:
        similarities += neg_mask * negative_weights

    return categorical_crossentropy_loss(
        y_true_matrix, similarities, from_logits=True
    ).mean()


def aoe_total_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    angle_tau: float = 20.0,
    cl_scale: float = 20.0,
    w_angle: float = 0.02,
    w_cl: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combine AnglE ranking loss with supervised contrastive loss."""
    # y_true comes in as [batch, 1] or [batch], flatten it
    y_true_flat = y_true.view(-1)
    
    angle_term = angle_loss(y_true, y_pred, tau=angle_tau, pooling="sum")

    # Use the new in-batch negative loss
    # Note: cl_scale corresponds to tau in the new function
    cl_term = in_batch_negative_loss(y_true_flat, y_pred, tau=cl_scale)

    total = w_angle * angle_term + w_cl * cl_term
    stats = {
        "angle_loss": float(angle_term.item()),
        "contrastive_loss": float(cl_term.item()),
        "total_loss": float(total.item()),
    }
    return total, stats