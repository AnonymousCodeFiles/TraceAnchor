from __future__ import annotations

from torch import nn

from traceanchor.detector.common import CausalTCN


class HostTCN(CausalTCN):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        channels: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )
        super().__init__(
            projection,
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )


__all__ = ["HostTCN"]
