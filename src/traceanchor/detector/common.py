from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalResidualBlock(nn.Module):
    """One left-padded residual block with no access to future positions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.causal_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # values: [batch, time, channels]
        residual = values
        convolved = self.causal_conv(
            F.pad(values.transpose(1, 2), (self.left_padding, 0))
        )
        transformed = self.pointwise(self.dropout(self.activation(convolved)))
        return self.normalization(residual + transformed.transpose(1, 2))


class CausalTCN(nn.Module):
    def __init__(
        self,
        input_projection: nn.Module,
        channels: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = input_projection
        self.blocks = nn.ModuleList(
            [
                CausalResidualBlock(channels, kernel_size, dilation, dropout)
                for dilation in dilations
            ]
        )
        self.output_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, 1),
        )

    def forward_sequence(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(values)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_head(hidden).squeeze(-1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(values)[:, -1]


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """Class-balanced focal BCE evaluated only where ``loss_mask`` is one."""

    targets = targets.to(dtype=logits.dtype)
    mask = loss_mask.to(dtype=logits.dtype)
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probability_of_target = torch.exp(-cross_entropy)
    alpha_for_target = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    losses = (
        alpha_for_target
        * (1.0 - probability_of_target).pow(gamma)
        * cross_entropy
        * mask
    )
    denominator = mask.sum()
    if denominator.item() == 0:
        return logits.sum() * 0.0
    return losses.sum() / denominator


__all__ = ["CausalResidualBlock", "CausalTCN", "focal_bce_with_logits"]
