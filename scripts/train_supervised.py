from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from pcr_ssl.data import RecipeCollator, RecipeDataset
from pcr_ssl.losses import masked_huber_loss
from pcr_ssl.model import SPASPredictor


def train_supervised_epoch(
    model: SPASPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    model.train()
    total = 0.0

    for batch in loader:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        optimizer.zero_grad()
        y_pred = model(batch)
        loss = masked_huber_loss(y_pred, batch["y_value"], batch["y_mask"])
        loss.backward()
        optimizer.step()
        total += loss.item()

    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate_supervised(model: SPASPredictor, loader: DataLoader, device: str = "cpu") -> dict[str, float]:
    model.eval()
    preds, trues, masks = [], [], []
    for batch in loader:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        y_pred = model(batch)
        preds.append(y_pred.cpu())
        trues.append(batch["y_value"].cpu())
        masks.append(batch["y_mask"].cpu())

    y_pred = torch.cat(preds)
    y_true = torch.cat(trues)
    y_mask = torch.cat(masks)

    valid = y_mask > 0
    y_p = y_pred[valid]
    y_t = y_true[valid]

    mae = torch.mean(torch.abs(y_p - y_t)).item()
    rmse = torch.sqrt(torch.mean((y_p - y_t) ** 2)).item()
    var = torch.var(y_t)
    r2 = (1 - torch.mean((y_t - y_p) ** 2) / var.clamp_min(1e-6)).item()
    return {"mae": mae, "rmse": rmse, "r2": r2}


def build_supervised_loader(samples, batch_size: int = 32, shuffle: bool = True):
    return DataLoader(RecipeDataset(samples), batch_size=batch_size, shuffle=shuffle, collate_fn=RecipeCollator())
