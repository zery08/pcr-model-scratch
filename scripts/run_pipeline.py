from __future__ import annotations

import argparse
import random
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SPAS predictor with SSL pretrain + supervised fine-tune")
    parser.add_argument("--mode", choices=["ssl", "supervised", "both"], default="both")
    parser.add_argument("--ssl-epochs", type=int, default=3)
    parser.add_argument("--sup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = make_dummy_samples()
    train, valid, test = split_by_incoming_id(samples)
    stats = fit_normalization_stats(train)
    apply_normalization(train, stats)
    apply_normalization(valid, stats)
    apply_normalization(test, stats)

    model = SPASPredictor(vocab_size=128, param_dim=8, n_spas=10, n_wl=5, n_loc=16)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_supervised_loader(train, batch_size=args.batch_size)
    valid_loader = build_supervised_loader(valid, batch_size=args.batch_size, shuffle=False)
    test_loader = build_supervised_loader(test, batch_size=args.batch_size, shuffle=False)

    if args.mode in {"ssl", "both"}:
        ssl_loader = build_ssl_loader(train, batch_size=args.batch_size)
        optimizer_ssl = torch.optim.Adam(model.backbone.parameters(), lr=args.lr)
        for epoch in range(1, args.ssl_epochs + 1):
            ssl_loss = train_ssl_epoch(model.backbone, ssl_loader, optimizer_ssl)
            print(f"[SSL] epoch={epoch} loss={ssl_loss:.4f}")
        torch.save(model.backbone.state_dict(), args.save_dir / "backbone_ssl.pt")
        print(f"[SSL] backbone checkpoint saved: {args.save_dir / 'backbone_ssl.pt'}")

    if args.mode in {"supervised", "both"}:
        if args.mode == "supervised" and (args.save_dir / "backbone_ssl.pt").exists():
            model.backbone.load_state_dict(torch.load(args.save_dir / "backbone_ssl.pt", map_location="cpu"))
            print(f"[Supervised] loaded pretrained backbone: {args.save_dir / 'backbone_ssl.pt'}")

        optimizer_ft = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.sup_epochs + 1):
            train_loss = train_supervised_epoch(model, train_loader, optimizer_ft)
            valid_metrics = evaluate_supervised(model, valid_loader)
            print(f"[Supervised] epoch={epoch} train_loss={train_loss:.4f} valid={valid_metrics}")

        test_metrics = evaluate_supervised(model, test_loader)
        print("[Fine-tuned] test:", test_metrics)
        torch.save(model.state_dict(), args.save_dir / "spas_predictor.pt")
        print(f"[Supervised] full model checkpoint saved: {args.save_dir / 'spas_predictor.pt'}")

    if not args.skip_baseline and args.mode != "ssl":
        baseline = SPASPredictor(vocab_size=128, param_dim=8, n_spas=10, n_wl=5, n_loc=16)
        optimizer_base = torch.optim.Adam(baseline.parameters(), lr=args.lr)
        for _ in range(args.sup_epochs):
            train_supervised_epoch(baseline, train_loader, optimizer_base)
        baseline_metrics = evaluate_supervised(baseline, test_loader)
        print("[Baseline(no SSL)] test:", baseline_metrics)


if __name__ == "__main__":
    main()
