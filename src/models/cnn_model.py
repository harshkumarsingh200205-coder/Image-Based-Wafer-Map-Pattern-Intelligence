"""Lightweight Convolutional Neural Network (WaferNet_Light) in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaferNetLight(nn.Module):
    """Lightweight 4-stage convolutional backbone for wafer map pattern recognition.

    Inputs: Tensor of shape (B, 1, 128, 128) or (B, 3, 128, 128)
    Outputs: Class logits of shape (B, num_classes)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 9):
        super().__init__()
        self.num_classes = num_classes

        # Stage 1: Initial feature extraction
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Stage 2: Spatial reduction
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Stage 3: Salient pattern layer (target for Grad-CAM)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Stage 4: High-level abstraction
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Final classification head
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # Global Average Pooling & Classification
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
