from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from pcr_ssl.data import RecipeCollator, RecipeDataset
from pcr_ssl.losses import info_nce_loss, msm_loss
from pcr_ssl.model import SSLRecipeBackbone


def random_mask(batch: dict[str, torch.Tensor], prob: float = 0.15) -> torch.Tensor:
    valid = batch["step_mask"] > 0
    return (torch.rand_like(batch["step_mask"]) < prob) & valid


def train_ssl_epoch(
    model: SSLRecipeBackbone,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_msm: float = 1.0,
    lambda_contrast: float = 1.0,
    device: str = "cpu",
) -> float:
    model.train()
    total = 0.0

    for batch in loader:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        optimizer.zero_grad()
        z_recipe, z_steps = model(batch)

        masked_positions = random_mask(batch).float()
        msm_pred = model.decode_msm(z_steps)
        loss_msm = msm_loss(
            msm_pred,
            batch["state_value"],
            batch["step_global_pos"],
            batch["param_mat"],
            masked_positions,
        )

        aug_batch = dict(batch)
        aug_batch["state_value"] = batch["state_value"] + 0.01 * torch.randn_like(batch["state_value"])
        z_recipe_aug, _ = model(aug_batch)
        loss_contrast = info_nce_loss(z_recipe, z_recipe_aug)

        loss = lambda_msm * loss_msm + lambda_contrast * loss_contrast
        loss.backward()
        optimizer.step()
        total += loss.item()

    return total / max(len(loader), 1)


def build_ssl_loader(samples, batch_size: int = 32):
    return DataLoader(RecipeDataset(samples), batch_size=batch_size, shuffle=True, collate_fn=RecipeCollator())
