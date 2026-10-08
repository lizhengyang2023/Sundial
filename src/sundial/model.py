"""Sundial Base: continuous patches, causal Transformer and conditional flow matching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SundialConfig:
    patch_size: int = 16
    max_context: int = 2880
    horizon: int = 720
    layers: int = 12
    dim: int = 768
    ff_dim: int = 3072
    heads: int = 12
    flow_dim: int = 768
    flow_layers: int = 3
    flow_steps: int = 50
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.dim % self.heads or (self.dim // self.heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.horizon < self.patch_size or self.max_context < self.patch_size:
            raise ValueError("horizon and max_context must be at least one patch")

    def to_dict(self) -> dict:
        return asdict(self)


def _rotate_half(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def _rope(x: Tensor) -> Tensor:
    # x: [batch, heads, tokens, head_dim]
    half = x.shape[-1] // 2
    inv = 1.0 / (10000 ** (torch.arange(half, device=x.device).float() / half))
    pos = torch.arange(x.shape[-2], device=x.device).float()
    phase = torch.outer(pos, inv)
    phase = torch.cat((phase, phase), dim=-1).to(x.dtype)[None, None]
    return x * phase.cos() + _rotate_half(x) * phase.sin()


class PatchEmbed(nn.Module):
    def __init__(self, cfg: SundialConfig):
        super().__init__()
        self.fc1 = nn.Linear(2 * cfg.patch_size, cfg.ff_dim)
        self.fc2 = nn.Linear(cfg.ff_dim, cfg.dim)
        self.skip = nn.Linear(2 * cfg.patch_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.patch_size = cfg.patch_size

    def forward(self, values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        pad = (-values.shape[-1]) % self.patch_size
        values = F.pad(values, (pad, 0))
        valid = F.pad(valid, (pad, 0))
        x = torch.cat((values.unfold(1, self.patch_size, self.patch_size),
                       valid.unfold(1, self.patch_size, self.patch_size)), dim=-1)
        tokens = self.skip(x) + self.drop(self.fc2(F.silu(self.fc1(x))))
        patch_valid = valid.unfold(1, self.patch_size, self.patch_size).any(-1)
        return tokens, patch_valid


class DecoderBlock(nn.Module):
    def __init__(self, cfg: SundialConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.q = nn.Linear(cfg.dim, cfg.dim)
        self.k = nn.Linear(cfg.dim, cfg.dim)
        self.v = nn.Linear(cfg.dim, cfg.dim)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.gate = nn.Linear(cfg.dim, cfg.ff_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ff_dim, bias=False)
        self.down = nn.Linear(cfg.ff_dim, cfg.dim, bias=False)
        self.heads = cfg.heads
        self.drop = cfg.dropout

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        b, n, d = x.shape
        z = self.norm1(x)
        split = lambda layer: layer(z).view(b, n, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = _rope(split(self.q)), _rope(split(self.k)), split(self.v)
        # True in SDPA means visible. The key mask excludes left padding.
        causal = torch.ones((n, n), device=x.device, dtype=torch.bool).tril()
        visible = causal[None, None] & mask[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=visible,
                                            dropout_p=self.drop if self.training else 0.0)
        x = x + self.out(a.transpose(1, 2).reshape(b, n, d))
        z = self.norm2(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


def _time_embedding(t: Tensor, dim: int = 256) -> Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freq[None]
    return torch.cat((args.cos(), args.sin()), dim=-1)


class AdaLNBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 3 * width))
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class FlowNet(nn.Module):
    def __init__(self, cfg: SundialConfig):
        super().__init__()
        self.input = nn.Linear(cfg.horizon, cfg.flow_dim)
        self.time = nn.Sequential(nn.Linear(256, cfg.flow_dim), nn.SiLU(),
                                  nn.Linear(cfg.flow_dim, cfg.flow_dim))
        self.condition = nn.Linear(cfg.dim, cfg.flow_dim)
        self.blocks = nn.ModuleList(AdaLNBlock(cfg.flow_dim) for _ in range(cfg.flow_layers))
        self.final_norm = nn.LayerNorm(cfg.flow_dim, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(cfg.flow_dim, 2 * cfg.flow_dim))
        self.output = nn.Linear(cfg.flow_dim, cfg.horizon)

    def forward(self, y_t: Tensor, t: Tensor, h: Tensor) -> Tensor:
        c = self.time(_time_embedding(t * 1000)) + self.condition(h)
        x = self.input(y_t)
        for block in self.blocks:
            x = block(x, c)
        shift, scale = self.final_mod(c).chunk(2, dim=-1)
        return self.output(self.final_norm(x) * (1 + scale) + shift)


class Sundial(nn.Module):
    def __init__(self, cfg: SundialConfig = SundialConfig()):
        super().__init__()
        self.cfg = cfg
        self.patch = PatchEmbed(cfg)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.dim)
        self.flow = FlowNet(cfg)

    @staticmethod
    def normalize(context: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        count = valid.sum(-1, keepdim=True).clamp_min(1)
        observed = torch.where(valid, context, torch.zeros_like(context))
        mean = observed.sum(-1, keepdim=True) / count
        var = torch.where(valid, (context - mean).square(), torch.zeros_like(context)).sum(-1, keepdim=True) / count
        std = var.sqrt().clamp_min(1e-4)
        normed = torch.where(valid, (context - mean) / std, torch.zeros_like(context))
        return normed, mean, std

    def encode(self, context: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if context.ndim != 2 or context.shape != valid.shape:
            raise ValueError("context and valid must both have shape [batch, time]")
        if context.shape[1] > self.cfg.max_context:
            context, valid = context[:, -self.cfg.max_context:], valid[:, -self.cfg.max_context:]
        x, mean, std = self.normalize(context, valid)
        h, patch_valid = self.patch(x, valid)
        for block in self.blocks:
            h = block(h, patch_valid)
        return self.norm(h), patch_valid, mean, std

    def training_loss(self, context: Tensor, context_valid: Tensor,
                      full_values: Tensor, full_valid: Tensor) -> Tensor:
        """One random flow time/noise per eligible (batch, patch) target."""
        p, f = self.cfg.patch_size, self.cfg.horizon
        h, patch_valid, mean, std = self.encode(context, context_valid)
        n = h.shape[1]
        needed = n * p + f
        if full_values.shape[-1] < needed:
            full_values = F.pad(full_values, (0, needed - full_values.shape[-1]))
            full_valid = F.pad(full_valid, (0, needed - full_valid.shape[-1]))
        targets = full_values[:, p:].unfold(1, f, p)[:, :n]
        target_mask = full_valid[:, p:].unfold(1, f, p)[:, :n]
        eligible = patch_valid & target_mask.any(-1)
        if not eligible.any():
            raise ValueError("batch has no context patch with an observed future")
        y = ((targets - mean[:, None]) / std[:, None])[eligible].clamp(-1, 1)
        m = target_mask[eligible]
        y = torch.where(m, y, torch.zeros_like(y))
        z = torch.randn_like(y)
        t = torch.rand((y.shape[0],), device=y.device, dtype=y.dtype)
        y_t = t[:, None] * y + (1 - t[:, None]) * z
        velocity = self.flow(y_t, t, h[eligible])
        # Sundial paper Eq. (6)-(8): u_theta(y_t,t,h) approximates y - epsilon.
        per_patch = ((velocity - (y - z)).square() * m).sum(-1) / m.sum(-1).clamp_min(1)
        # Keep each chosen variable's weight equal even when context lengths differ.
        batch_index = eligible.nonzero(as_tuple=True)[0]
        per_series = torch.zeros(context.shape[0], device=y.device, dtype=per_patch.dtype)
        per_series = per_series.scatter_add(0, batch_index, per_patch)
        return (per_series / eligible.sum(-1).clamp_min(1)).mean()

    @torch.no_grad()
    def forecast(self, context: Tensor, horizon: int, samples: int = 1,
                 valid: Tensor | None = None, steps: int | None = None) -> Tensor:
        """Return [batch, samples, horizon] in the input's original units."""
        if horizon < 1 or samples < 1:
            raise ValueError("horizon and samples must be positive")
        if valid is None:
            valid = torch.ones_like(context, dtype=torch.bool)
        steps = self.cfg.flow_steps if steps is None else steps
        if steps < 1:
            raise ValueError("steps must be positive")
        history = context
        outputs: list[Tensor] = []
        while sum(chunk.shape[-1] for chunk in outputs) < horizon:
            h, mask, mean, std = self.encode(history, valid)
            # Context is left padded: the latest valid token is always the last one.
            last = torch.full((h.shape[0],), h.shape[1] - 1, device=h.device, dtype=torch.long)
            condition = h[torch.arange(h.shape[0], device=h.device), last]
            condition = condition.repeat_interleave(samples, dim=0)
            x = torch.randn((condition.shape[0], self.cfg.horizon),
                            device=context.device, dtype=context.dtype)
            for k in range(steps):
                t = x.new_full((x.shape[0],), k / steps)
                x = x + self.flow(x, t, condition) / steps
            pred = x.reshape(history.shape[0], samples, -1) * std[:, None] + mean[:, None]
            outputs.append(pred)
            # Roll the mean into history; for >F forecasts this is an explicit
            # deterministic continuation choice, not a fresh sample trajectory.
            if sum(chunk.shape[-1] for chunk in outputs) < horizon:
                history = torch.cat((history, pred.mean(1)), dim=-1)[:, -self.cfg.max_context:]
                valid = torch.ones_like(history, dtype=torch.bool)
        return torch.cat(outputs, dim=-1)[..., :horizon]
