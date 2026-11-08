"""PyTorch neural network architecture for SignBridge isolated sign recognition."""

from __future__ import annotations

import torch
import torch.nn as nn


class WordSignModel(nn.Module):
    """1D CNN + Bidirectional GRU model for isolated ASL sign recognition.

    Inputs:
        x: Tensor of shape (batch, T=32, N_FEATURES=104).

    Outputs:
        logits: Tensor of shape (batch, num_classes=30).
    """

    def __init__(
        self,
        in_features: int = 104,
        num_classes: int = 30,
        cnn_channels: tuple[int, ...] = (128, 128),
        kernel_size: int = 3,
        gru_hidden_size: int = 128,
        num_gru_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.gru_hidden_size = gru_hidden_size

        # Conv1d expects (batch, in_channels, time)
        conv_layers: list[nn.Module] = []
        prev_ch = in_features
        for ch in cnn_channels:
            conv_layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2.0),
            ])
            prev_ch = ch

        self.conv = nn.Sequential(*conv_layers)

        # BiGRU
        self.gru = nn.GRU(
            input_size=prev_ch,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )

        # Classifier head:
        # BiGRU produces 2 * gru_hidden_size per step
        # We concatenate mean pooling and max pooling over time -> 4 * gru_hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * gru_hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, T, F) -> (B, 32, 104)

        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        # (B, T, F) -> (B, F, T) for 1D convolution
        x_conv = x.transpose(1, 2)
        conv_out = self.conv(x_conv)  # (B, C, T)

        # (B, C, T) -> (B, T, C) for GRU
        gru_in = conv_out.transpose(1, 2)
        gru_out, _ = self.gru(gru_in)  # (B, T, 2 * H)

        # Temporal pooling: mean pool + max pool
        mean_pool = torch.mean(gru_out, dim=1)  # (B, 2 * H)
        max_pool, _ = torch.max(gru_out, dim=1)  # (B, 2 * H)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, 4 * H)

        logits = self.fc(pooled)  # (B, num_classes)
        return logits
