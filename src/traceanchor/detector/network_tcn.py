from __future__ import annotations

from torch import nn

from traceanchor.detector.common import CausalTCN


class NetworkTCN(CausalTCN):
    def __init__(
        self,
        input_dim: int,
        *,
        channels: int = 64,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            nn.Linear(input_dim, channels),
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )


__all__ = ["NetworkTCN"]
