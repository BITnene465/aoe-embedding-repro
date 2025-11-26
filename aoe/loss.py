"""AoE / AnglE loss implementations matching the Official Repository."""

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
    """Compute the AnglE ranking loss matching official implementation."""
    # Official Repo: checks dimensions strictly
    if y_pred.dim() != 2:
        raise ValueError("y_pred must be a 2D tensor [batch, feat]")
    
    batch_size, feat_dim = y_pred.shape
    y_true_flat = y_true.view(-1)
    
    # Extract pairwise scores (Zigzag assumption: s1, s1, s2, s2...)
    pair_scores = y_true_flat[0::2].float()
    
    # Ranking Matrix: 1 if i > j
    order = (pair_scores[:, None] > pair_scores[None, :]).float()

    # Split Real and Imag
    dim = feat_dim // 2
    pred_re = y_pred[:, :dim]
    pred_im = y_pred[:, dim:]

    # Extract 'a', 'b', 'c', 'd' matching official complex division logic
    # a, b: query (odd indices); c, d: key (even indices)
    # Note: Official repo logic assumes input [q1, k1, q2, k2...]
    a = pred_re[0::2]
    b = pred_im[0::2]
    c = pred_re[1::2]
    d = pred_im[1::2]

    # Complex Division: (a+bi)/(c+di)
    # Official Repo implementation details:
    # z = c^2 + d^2
    z = torch.sum(c**2 + d**2, dim=1, keepdim=True).clamp(min=1e-9)
    
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    # Normalize by magnitude ratio (Official feature)
    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True).clamp(min=1e-9)**0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True).clamp(min=1e-9)**0.5
    re /= (dz / dw)
    im /= (dz / dw)

    # Official Pooling: Cat -> Sum/Mean (NO atan2)
    y_pred_angle = torch.cat((re, im), dim=1)
    
    if pooling == "sum":
        pooled = torch.sum(y_pred_angle, dim=1)
    elif pooling == "mean":
        pooled = torch.mean(y_pred_angle, dim=1)
    else:
        pooled = torch.sum(y_pred_angle, dim=1)

    scores = pooled * tau
    
    # CoSENT Ranking Logic
    # Minimize score_neg - score_pos -> Maximize score_pos - score_neg
    diff = scores[None, :] - scores[:, None]
    
    # Official masking logic
    masked_diff = diff - (1.0 - order) * 1e12
    
    # Check valid pairs
    if order.sum() == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    # LogSumExp
    flat = masked_diff.reshape(-1)
    flat = torch.cat([flat.new_zeros(1, device=flat.device, dtype=flat.dtype), flat], dim=0)
    return torch.logsumexp(flat, dim=0).to(torch.float32)


def categorical_crossentropy_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    from_logits: bool = True,
) -> torch.Tensor:
    """Compute categorical crossentropy matching official utils."""
    if from_logits:
        return -(F.log_softmax(y_pred, dim=1) * y_true).sum(dim=1)
    return -(torch.log(y_pred, dim=1) * y_true).sum(dim=1)


def make_target_matrix(y_true: torch.Tensor) -> torch.Tensor:
    """
    Construct the target matrix for in-batch negative loss.
    OFFICIAL IMPLEMENTATION: Uses .int() casting.
    This creates 'buckets' for STS scores (e.g., 3.2 and 3.8 become class '3').
    """
    device = y_true.device
    idxs = torch.arange(0, y_true.shape[0]).int().to(device)
    
    # STRICT OFFICIAL LOGIC: Cast to int
    y_true = y_true.int()
    
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]

    y_true_row = y_true[None, :]
    y_true_col = y_true[:, None]

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
    """
    Supervised Contrastive Loss matching Official Repo.
    Uses label-based masking via make_target_matrix (with int casting).
    """
    device = y_true.device
    y_true = y_true.view(-1)
    
    # Official Repo: Generates masks based on 'int' labels
    neg_mask = make_target_matrix(y_true == 0)
    y_true_matrix = make_target_matrix(y_true)

    # Similarity calculation
    y_pred = F.normalize(y_pred, dim=1, p=2)
    similarities = y_pred @ y_pred.T
    similarities = similarities - torch.eye(y_pred.shape[0]).to(device) * 1e12
    similarities = similarities * tau

    if negative_weights > 0:
        similarities += neg_mask * negative_weights

    # Cross Entropy
    return categorical_crossentropy_loss(
        y_true_matrix, similarities, from_logits=True
    ).mean()


def cosine_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    tau: float = 20.0,
) -> torch.Tensor:
    """Optional Cosine MSE Loss."""
    y_pred = F.normalize(y_pred, p=2, dim=1)
    batch_size = y_pred.shape[0]
    pairs = y_pred.view(batch_size // 2, 2, -1)
    cos_sim = (pairs[:, 0] * pairs[:, 1]).sum(dim=-1)
    
    targets = y_true.view(-1)[0::2]
    # Official repo doesn't enforce dynamic norm inside loss, keeps it simple
    return F.mse_loss(cos_sim, targets)


def aoe_total_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    angle_tau: float = 20.0,
    cl_scale: float = 20.0,
    w_angle: float = 0.02, # Default for fine-tuning
    w_cl: float = 1.0,     # Default for fine-tuning
    w_cosine: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    y_true_flat = y_true.view(-1)
    
    # 1. Angle Loss (Main STS objective)
    loss_angle = angle_loss(y_true_flat, y_pred, tau=angle_tau, pooling="sum")
    
    # 2. Contrastive Loss (Auxiliary 'Bucket' objective)
    loss_cl = in_batch_negative_loss(y_true_flat, y_pred, tau=cl_scale)
    
    # 3. Cosine Loss
    loss_cosine = torch.tensor(0.0, device=y_pred.device)
    if w_cosine > 0:
        # Simple norm if needed
        targets = y_true_flat
        if targets.max() > 1.1:
            targets = targets / 5.0
        loss_cosine = cosine_loss(targets, y_pred)

    total_loss = w_angle * loss_angle + w_cl * loss_cl + w_cosine * loss_cosine
    
    stats = {
        "angle_loss": loss_angle.item(),
        "contrastive_loss": loss_cl.item(),
        "cosine_loss": loss_cosine.item(),
        "total_loss": total_loss.item()
    }
    
    return total_loss, stats