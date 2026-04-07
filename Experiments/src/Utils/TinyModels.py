import math
import torch
from torch import nn
import numpy as np


class DiscreteTimeResidualBlock(nn.Module):
    """Generic block to learn a nonlinear function f(x, t), where
    t is discrete and x is continuous."""

    def __init__(self, d_model: int, maxlen: int = 512):
        super().__init__()
        self.d_model = d_model
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.norm(x + self.lin2(self.act(self.lin1(x))))


class BasicDiscreteTimeModel(nn.Module):
    def __init__(self, d: int, d_model: int = 128, n_layers: int = 3):
        super().__init__()
        self.d = d
        self.d_model = d_model
        self.n_layers = n_layers
        self.lin_in = nn.Linear(d, d_model)
        self.lin_out = nn.Linear(d_model, d)
        self.blocks = nn.ParameterList(
            [DiscreteTimeResidualBlock(d_model=d_model) for _ in range(n_layers)]
        )

        # Optional: initialize weights if desired
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.lin_in(x)
        for block in self.blocks:
            x = block(x)
        return self.lin_out(x)


class ContinuousTimeEmbedding(nn.Module):
    """Sinusoidal + MLP time embedding for continuous t."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (batch,) or (batch, 1) float tensor, e.g. in [0, 1] or [0, T].
        Returns: (batch, d_model)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)
        half_dim = self.d_model // 2
        device = t.device
        freqs = torch.exp(
            torch.arange(half_dim, device=device)
            * (-math.log(10000.0) / (half_dim - 1))
        )  # (half_dim,)
        args = t * freqs  # (B, half_dim)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, 2*half_dim)
        if emb.size(-1) < self.d_model:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class SimpleResidualBlock(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.time_emb = ContinuousTimeEmbedding(d_model=d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, d_model)
        t: (batch,) or (batch, 1) float tensor
        """
        out = x + self.time_emb(t)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        return self.norm(x + out)


class SimpleTimeModel(nn.Module):
    """
    Time‑conditioned MLP mapping R^{d_in} -> R^{d_out}.

    For the GMM experiment:
      - d_out = data dimension (e.g. 8)
      - n_channels = model_order (e.g. 2)
      - d_in = n_channels * d_out (e.g. 16)
    """

    def __init__(self, d_in: int, d_out: int, d_model: int = 128, n_blocks: int = 3):
        super().__init__()
        self.d_in = d_in       # flattened input size (e.g. 16)
        self.d_out = d_out     # output size (e.g. 8)
        self.d_model = d_model
        self.n_blocks = n_blocks

        self.in_layer = nn.Linear(d_in, d_model)
        self.out_layer = nn.Linear(d_model, d_out)
        self.blocks = nn.ParameterList(
            [SimpleResidualBlock(d_model=d_model) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, d_in)
        t: (batch,) or (batch, 1) float tensor
        """
        x = self.in_layer(x)
        for block in self.blocks:
            x = block(x, t)
        return self.out_layer(x)
