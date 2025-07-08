import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableBlur(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(1, 1, 3, 3))
        self.channels = channels

    def forward(self, x):
        # Normalize kernel to sum to 1 (like Gaussian blur)
        self.kernel.data = self.kernel.data / (self.kernel.data.sum() + 1e-8)
        k = self.kernel / (self.kernel.sum() + 1e-8)
        
        k = k.expand(self.channels, 1, 3, 3)
        return F.conv2d(x, k, padding=1, groups=self.channels)
