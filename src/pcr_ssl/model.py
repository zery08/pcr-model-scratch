from __future__ import annotations

import torch
from torch import nn


class SSLRecipeBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        param_dim: int,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.step_embed = nn.Embedding(vocab_size, hidden_dim)
        self.state_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.param_proj = nn.Sequential(nn.Linear(param_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.pos_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.msm_head = nn.Linear(hidden_dim, param_dim + 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        step = self.step_embed(batch["step_name"])
        state = self.state_proj(batch["state_value"].unsqueeze(-1))
        param = self.param_proj(batch["param_mat"])
        pos = self.pos_proj(batch["step_global_pos"].unsqueeze(-1))

        h = self.norm(step + state + param + pos)
        key_padding_mask = batch["step_mask"] == 0
        z_steps = self.encoder(h, src_key_padding_mask=key_padding_mask)

        masked = batch["step_mask"].unsqueeze(-1)
        z_recipe = (z_steps * masked).sum(dim=1) / (masked.sum(dim=1).clamp_min(1.0))
        return z_recipe, z_steps

    def decode_msm(self, z_steps: torch.Tensor) -> torch.Tensor:
        return self.msm_head(z_steps)


class ConditionEncoder(nn.Module):
    def __init__(self, n_spas: int, n_wl: int, n_loc: int, hidden_dim: int = 128):
        super().__init__()
        self.spas_emb = nn.Embedding(n_spas, hidden_dim)
        self.wl_emb = nn.Embedding(n_wl, hidden_dim)
        self.loc_emb = nn.Embedding(n_loc, hidden_dim)
        self.xy_proj = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU())

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        z = torch.cat(
            [
                self.spas_emb(batch["spas_item_id"]),
                self.wl_emb(batch["wl_id"]),
                self.loc_emb(batch["wf_loc_id"]),
                self.xy_proj(torch.stack([batch["wf_loc_x"], batch["wf_loc_y"]], dim=-1)),
            ],
            dim=-1,
        )
        return self.out(z)


class SPASPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        param_dim: int,
        n_spas: int,
        n_wl: int,
        n_loc: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.backbone = SSLRecipeBackbone(vocab_size=vocab_size, param_dim=param_dim, hidden_dim=hidden_dim)
        self.condition = ConditionEncoder(n_spas=n_spas, n_wl=n_wl, n_loc=n_loc, hidden_dim=hidden_dim)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        z_recipe, _ = self.backbone(batch)
        z_cond = self.condition(batch)
        y_pred = self.reg_head(torch.cat([z_recipe, z_cond], dim=-1)).squeeze(-1)
        return y_pred
