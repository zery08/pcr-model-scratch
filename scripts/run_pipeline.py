from __future__ import annotations

import random

import torch

from pcr_ssl.data import RecipeSample, apply_normalization, fit_normalization_stats, split_by_incoming_id
from pcr_ssl.model import SPASPredictor
from scripts.train_ssl import build_ssl_loader, train_ssl_epoch
from scripts.train_supervised import build_supervised_loader, evaluate_supervised, train_supervised_epoch


def make_dummy_samples(n: int = 200, seq_len: int = 10, param_dim: int = 8):
    samples = []
    for i in range(n):
        incoming = f"INC_{i // 2}"
        step_name = [random.randint(1, 99) for _ in range(seq_len)]
        state_value = [random.random() for _ in range(seq_len)]
        step_global_pos = [j / seq_len for j in range(seq_len)]
        param_mat = [[random.random() for _ in range(param_dim)] for _ in range(seq_len)]
        wf_x = random.uniform(-1, 1)
        wf_y = random.uniform(-1, 1)
        y = sum(state_value) / seq_len + 0.1 * wf_x - 0.05 * wf_y
        samples.append(
            RecipeSample(
                incoming_id=incoming,
                step_name=step_name,
                state_value=state_value,
                step_global_pos=step_global_pos,
                param_mat=param_mat,
                spas_item_id=random.randint(0, 9),
                wl_id=random.randint(0, 4),
                wf_loc_id=random.randint(0, 15),
                wf_loc_x=wf_x,
                wf_loc_y=wf_y,
                y_value=y,
                y_mask=1.0,
            )
        )
    return samples


def main() -> None:
    random.seed(7)
    torch.manual_seed(7)

    samples = make_dummy_samples()
    train, valid, test = split_by_incoming_id(samples)
    stats = fit_normalization_stats(train)
    apply_normalization(train, stats)
    apply_normalization(valid, stats)
    apply_normalization(test, stats)

    model = SPASPredictor(vocab_size=128, param_dim=8, n_spas=10, n_wl=5, n_loc=16)

    ssl_loader = build_ssl_loader(train, batch_size=16)
    optimizer_ssl = torch.optim.Adam(model.backbone.parameters(), lr=1e-3)
    ssl_loss = train_ssl_epoch(model.backbone, ssl_loader, optimizer_ssl)

    train_loader = build_supervised_loader(train, batch_size=16)
    valid_loader = build_supervised_loader(valid, batch_size=16, shuffle=False)
    test_loader = build_supervised_loader(test, batch_size=16, shuffle=False)

    optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        train_supervised_epoch(model, train_loader, optimizer_ft)

    valid_metrics = evaluate_supervised(model, valid_loader)
    test_metrics = evaluate_supervised(model, test_loader)

    baseline = SPASPredictor(vocab_size=128, param_dim=8, n_spas=10, n_wl=5, n_loc=16)
    optimizer_base = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    for _ in range(3):
        train_supervised_epoch(baseline, train_loader, optimizer_base)
    baseline_metrics = evaluate_supervised(baseline, test_loader)

    print("[SSL pretrain] loss:", round(ssl_loss, 4))
    print("[Fine-tuned] valid:", valid_metrics)
    print("[Fine-tuned] test:", test_metrics)
    print("[Baseline(no SSL)] test:", baseline_metrics)


if __name__ == "__main__":
    main()
