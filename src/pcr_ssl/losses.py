from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, y_mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    raw = F.huber_loss(y_pred, y_true, delta=delta, reduction="none")
    masked = raw * y_mask
    return masked.sum() / y_mask.sum().clamp_min(1.0)


def masked_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
    raw = F.mse_loss(y_pred, y_true, reduction="none")
    masked = raw * y_mask
    return masked.sum() / y_mask.sum().clamp_min(1.0)


def msm_loss(
    msm_pred: torch.Tensor,
    state_value: torch.Tensor,
    step_global_pos: torch.Tensor,
    param_mat: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    target = torch.cat([state_value.unsqueeze(-1), step_global_pos.unsqueeze(-1), param_mat], dim=-1)
    raw = F.mse_loss(msm_pred, target, reduction="none").mean(dim=-1)
    masked = raw * mask
    return masked.sum() / mask.sum().clamp_min(1.0)


def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.t() / temperature
    labels = torch.arange(z_a.size(0), device=z_a.device)
    return F.cross_entropy(logits, labels)
