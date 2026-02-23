from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class RecipeSample:
    incoming_id: str
    step_name: Sequence[int]
    state_value: Sequence[float]
    step_global_pos: Sequence[float]
    param_mat: Sequence[Sequence[float]]
    spas_item_id: int
    wl_id: int
    wf_loc_id: int
    wf_loc_x: float
    wf_loc_y: float
    y_value: float
    y_mask: float = 1.0


class RecipeDataset(Dataset):
    def __init__(self, samples: Sequence[RecipeSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RecipeSample:
        return self.samples[idx]


class RecipeCollator:
    def __init__(self, pad_value: int = 0):
        self.pad_value = pad_value

    def __call__(self, batch: Sequence[RecipeSample]) -> Dict[str, torch.Tensor]:
        max_steps = max(len(item.step_name) for item in batch)
        param_dim = len(batch[0].param_mat[0])

        step_name = torch.full((len(batch), max_steps), self.pad_value, dtype=torch.long)
        state_value = torch.zeros((len(batch), max_steps), dtype=torch.float32)
        step_global_pos = torch.zeros((len(batch), max_steps), dtype=torch.float32)
        param_mat = torch.zeros((len(batch), max_steps, param_dim), dtype=torch.float32)
        step_mask = torch.zeros((len(batch), max_steps), dtype=torch.float32)

        for i, item in enumerate(batch):
            n = len(item.step_name)
            step_name[i, :n] = torch.tensor(item.step_name, dtype=torch.long)
            state_value[i, :n] = torch.tensor(item.state_value, dtype=torch.float32)
            step_global_pos[i, :n] = torch.tensor(item.step_global_pos, dtype=torch.float32)
            param_mat[i, :n] = torch.tensor(item.param_mat, dtype=torch.float32)
            step_mask[i, :n] = 1.0

        return {
            "incoming_id": [item.incoming_id for item in batch],
            "step_name": step_name,
            "state_value": state_value,
            "step_global_pos": step_global_pos,
            "param_mat": param_mat,
            "step_mask": step_mask,
            "spas_item_id": torch.tensor([item.spas_item_id for item in batch], dtype=torch.long),
            "wl_id": torch.tensor([item.wl_id for item in batch], dtype=torch.long),
            "wf_loc_id": torch.tensor([item.wf_loc_id for item in batch], dtype=torch.long),
            "wf_loc_x": torch.tensor([item.wf_loc_x for item in batch], dtype=torch.float32),
            "wf_loc_y": torch.tensor([item.wf_loc_y for item in batch], dtype=torch.float32),
            "y_value": torch.tensor([item.y_value for item in batch], dtype=torch.float32),
            "y_mask": torch.tensor([item.y_mask for item in batch], dtype=torch.float32),
        }


def split_by_incoming_id(
    samples: Sequence[RecipeSample],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[List[RecipeSample], List[RecipeSample], List[RecipeSample]]:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1")

    incoming_ids = sorted({sample.incoming_id for sample in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(incoming_ids)

    n_train = int(len(incoming_ids) * ratios[0])
    n_valid = int(len(incoming_ids) * ratios[1])

    train_ids = set(incoming_ids[:n_train])
    valid_ids = set(incoming_ids[n_train : n_train + n_valid])
    test_ids = set(incoming_ids[n_train + n_valid :])

    train = [sample for sample in samples if sample.incoming_id in train_ids]
    valid = [sample for sample in samples if sample.incoming_id in valid_ids]
    test = [sample for sample in samples if sample.incoming_id in test_ids]
    return train, valid, test


def fit_normalization_stats(samples: Iterable[RecipeSample]) -> Dict[str, Tuple[float, float]]:
    state_values, wf_loc_x, wf_loc_y = [], [], []
    param_rows = []
    for sample in samples:
        state_values.extend(sample.state_value)
        wf_loc_x.append(sample.wf_loc_x)
        wf_loc_y.append(sample.wf_loc_y)
        param_rows.extend(sample.param_mat)

    params = np.array(param_rows, dtype=np.float32)
    return {
        "state_value": _safe_mean_std(np.array(state_values, dtype=np.float32)),
        "wf_loc_x": _safe_mean_std(np.array(wf_loc_x, dtype=np.float32)),
        "wf_loc_y": _safe_mean_std(np.array(wf_loc_y, dtype=np.float32)),
        "param_mat": (float(params.mean()), float(params.std() + 1e-6)),
    }


def apply_normalization(samples: Iterable[RecipeSample], stats: Dict[str, Tuple[float, float]]) -> None:
    state_mean, state_std = stats["state_value"]
    x_mean, x_std = stats["wf_loc_x"]
    y_mean, y_std = stats["wf_loc_y"]
    p_mean, p_std = stats["param_mat"]

    for sample in samples:
        sample.state_value = [(v - state_mean) / state_std for v in sample.state_value]
        sample.wf_loc_x = (sample.wf_loc_x - x_mean) / x_std
        sample.wf_loc_y = (sample.wf_loc_y - y_mean) / y_std
        sample.param_mat = [[(v - p_mean) / p_std for v in row] for row in sample.param_mat]


def _safe_mean_std(values: np.ndarray) -> Tuple[float, float]:
    return float(values.mean()), float(values.std() + 1e-6)
