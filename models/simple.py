import torch.nn as nn
from .filters import (
    LearnableBlur,
    LearnableBoxBlur,
    BaseFilter,
    GuidedFilter,
    IdentityFilter,
    # FastGuidedFilter,
    LearnableEdgeBlur,
)


class SimpleCNN(nn.Module):
    def __init__(self, D: int = 32, blur_type: str = "none", use_skip: bool = True, **blur_kwargs):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, D, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(D, 3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(D)
        self.blur = build_blur(blur_type, **blur_kwargs)
        self.use_skip = use_skip
        self._init_weights()

    def forward(self, img):
        feat = self.relu(self.bn1(self.conv1(img)))
        residual = self.conv2(feat)
        blurred = self.blur(residual, img)

        if self.use_skip:
            return blurred + img
        else:
            return blurred

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


def build_blur(blur_type: str, **kw) -> BaseFilter:
    blur_type = blur_type.lower()
    if blur_type == "none":
        return IdentityFilter()

    if blur_type == "blur":
        return LearnableBlur(channels=kw.get("channels", 3), k_size=kw.get("k_size", 3), mode=kw.get("mode", "softmax"), eps=kw.get("eps", 1e-8))

    if blur_type == "box":
        return LearnableBoxBlur(
            channels=kw.get("channels", 3), k_size=kw.get("k_size", 3), eps=kw.get("eps", 1e-8)
        )

    if blur_type == "edge":
        return LearnableEdgeBlur(
            channels=kw.get("channels", 3), k_size=kw.get("k_size", 5)
        )

    if blur_type == "guided":
        return GuidedFilter(radius=kw.get("radius", 5), eps=kw.get("eps", 1e-8))

    # if blur_type == "fast_guided":
    #     return FastGuidedFilter(radius=kw.get("radius", 5),
    #                             eps=kw.get("eps", 1e-8))

    raise ValueError(f"Unknown blur_type '{blur_type}'")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class DeepCNN(nn.Module):
    def __init__(self, D=64, num_blocks=4):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, D, 3, 1, 1), nn.BatchNorm2d(D), nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(D) for _ in range(num_blocks)])
        self.exit = nn.Conv2d(D, 3, 3, 1, 1)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        x = self.entry(img)
        x = self.res_blocks(x)
        return self.exit(x) + img
