import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableBlur(nn.Module):
    def __init__(self, channels: int = 3, k_size: int = 3):
        super().__init__()
        assert k_size % 2 == 1, "Kernel size must be odd."
        self.channels = channels
        self.k_size = k_size
        self.eps = 1e-8

        self.kernel = nn.Parameter(torch.randn(channels, 1, k_size, k_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel‑wise normalisation so each kernel sums to 1
        flat = self.kernel.view(self.channels, -1)
        norm = flat.sum(dim=1, keepdim=True).view(self.channels, 1, 1, 1)
        k = self.kernel / (norm + self.eps)

        padding = self.k_size // 2
        return F.conv2d(x, k, padding=padding, groups=self.channels)


class LearnableBoxBlur(nn.Module):
    def __init__(self, channels: int = 3, k_size: int = 3):
        super().__init__()
        assert k_size % 2 == 1, "Kernel size must be odd."
        self.channels = channels
        self.k_size = k_size
        self.eps = 1e-8

        kernel = torch.ones(channels, 1, k_size, k_size) / (k_size * k_size)
        self.weight = nn.Parameter(kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Re‑normalise so each kernel stays on the probability simplex
        flat = self.weight.view(self.channels, -1)
        norm = flat.sum(dim=1, keepdim=True).view(self.channels, 1, 1, 1)
        w = self.weight / (norm + self.eps)

        padding = self.k_size // 2
        return F.conv2d(x, w, padding=padding, groups=self.channels)
