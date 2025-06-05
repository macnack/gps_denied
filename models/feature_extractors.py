import torch
from torch import nn
from torchvision import models

class vgg16Conv(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()

        if model_path:
            print('Loading VGG16 from:', model_path)
            vgg16 = torch.load(model_path)
        else:
            print('Loading pretrained VGG16 from torchvision...', end='')
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            print('done')

        # Extract layers up to ReLU after 3rd conv block (index 15)
        self.features = nn.Sequential(*list(vgg16.features.children())[:15])

        # Freeze early layers (e.g., conv1 and conv2)
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d) and layer.out_channels < 256:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.features(x)