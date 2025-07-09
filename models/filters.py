import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

class BaseFilter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat: torch.Tensor, *extras: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this!")

class IdentityFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.m = nn.Identity()

    def forward(self, feat: torch.Tensor, *extras: torch.Tensor) -> torch.Tensor:
        return feat
        
class LearnableBlur(BaseFilter):
    """
    Gaussian blur with learnable kernel that normalizes the kernel to stay on the probability simplex.
    """

    def __init__(self, channels: int = 3, k_size: int = 3, mode: str = "softmax"):
        super().__init__()
        assert k_size % 2 == 1, "Kernel size must be odd."
        self.mode = mode
        self.channels = channels
        self.k_size = k_size
        self.eps = 1e-8

        self.kernel = nn.Parameter(torch.randn(channels, 1, k_size, k_size))

    def forward(self, feat: torch.Tensor, *extras) -> torch.Tensor:
        if self.mode == "softmax":
            k = F.softmax(self.kernel.view(feat.size(1), -1), dim=-1).view_as(self.kernel)
        else:
            flat = self.kernel.view(feat.size(1), -1)
            norm = flat.sum(dim=1, keepdim=True).view_as(self.kernel)
            k = self.kernel / (norm + self.eps)

        padding = self.k_size // 2
        return F.conv2d(feat, k, padding=padding, groups=feat.size(1))


class LearnableBoxBlur(BaseFilter):
    """
    Learnable box blur that normalizes the kernel to stay on the probability simplex.
    """

    def __init__(self, channels: int = 3, k_size: int = 3):
        super().__init__()
        assert k_size % 2 == 1, "Kernel size must be odd."
        self.channels = channels
        self.k_size = k_size
        self.eps = 1e-8

        kernel = torch.ones(channels, 1, k_size, k_size) / (k_size * k_size)
        self.weight = nn.Parameter(kernel)

    def forward(self, feat: torch.Tensor, *extras) -> torch.Tensor:
        flat = self.weight.view(self.channels, -1)
        norm = flat.sum(dim=1, keepdim=True).view(self.channels, 1, 1, 1)
        w = self.weight / (norm + self.eps)

        padding = self.k_size // 2
        return F.conv2d(feat, w, padding=padding, groups=self.channels)


class LearnableEdgeBlur(BaseFilter):
    """
    Learnable edge-aware blur that uses Sobel gradients to determine where to apply the blur.
    """

    def __init__(self, channels: int = 3, k_size: int = 5):
        super().__init__()
        assert k_size % 2 == 1, "Kernel size must be odd."
        self.blur = nn.Conv2d(
            channels,
            channels,
            kernel_size=k_size,
            padding=k_size // 2,
            groups=channels,
            bias=False,
        )
        # Initialize with a box blur
        self.blur.weight.data.fill_(1.0 / (k_size * k_size))

        # Trainable parameters for edge blending.
        self.logit_slope = nn.Parameter(torch.tensor(10.0))
        self.log_thresh = nn.Parameter(torch.logit(torch.tensor(0.05)))

    def forward(self, feat: torch.Tensor, *extras) -> torch.Tensor:
        grads = kornia.filters.sobel(feat)
        mag = torch.sqrt(grads[:, 0] ** 2 + grads[:, 1] ** 2 + 1e-8).mean(1, keepdim=True)

        slope = F.softplus(self.logit_slope)
        thresh = torch.sigmoid(self.log_thresh)

        alpha = torch.sigmoid(-slope * (mag - thresh))
        blur = self.blur(feat)
        return alpha * blur + (1 - alpha) * feat


class GuidedFilter(BaseFilter):
    """
    Differentiable guided filter using Kornia's guided_blur.
    """

    def __init__(self, radius: int = 5, eps: float = 1e-8):
        super().__init__()
        self.radius = radius
        self.eps = eps

    def forward(self, feat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        k = 2 * self.radius + 1
        return kornia.filters.guided_blur(
            img, feat, (k, k), self.eps, border_type="reflect", subsample=1
        )


# class FastGuidedFilter(BaseFilter):
#     """
#     Upsampling variant: low-res guided filter + bilinear upsample.
#     """

#     def __init__(self, radius: int = 5, eps: float = 1e-8):
#         super().__init__()
#         self.radius = radius
#         self.eps = eps

#     def forward(self, lr_x: torch.Tensor, lr_y: torch.Tensor, hr_x: torch.Tensor) -> torch.Tensor:
#         k = 2 * self.radius + 1
#         lr_out = kornia.filters.guided_blur(
#             lr_x, lr_y, (k, k), self.eps, border_type="reflect", subsample=1
#         )
#         hr_out = F.interpolate(
#             lr_out, size=hr_x.shape[2:], mode="bilinear", align_corners=False
#         )
#         return hr_out * hr_x
