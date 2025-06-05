import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientBatch(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel-like filters for x and y gradients
        wx = torch.tensor([[-0.5, 0, 0.5]], dtype=torch.float32).view(1, 1, 1, 3)
        wy = torch.tensor([[-0.5], [0], [0.5]], dtype=torch.float32).view(1, 1, 3, 1)
        self.register_buffer('wx', wx)
        self.register_buffer('wy', wy)
        self.padx = nn.ReplicationPad2d((1, 1, 0, 0))
        self.pady = nn.ReplicationPad2d((0, 0, 1, 1))

    def forward(self, img):
        batch_size, channels, height, width = img.size()
        img = img.view(batch_size * channels, 1, height, width)

        dx = F.conv2d(self.padx(img), self.wx).view(batch_size, channels, height, width)
        dy = F.conv2d(self.pady(img), self.wy).view(batch_size, channels, height, width)

        return dx, dy