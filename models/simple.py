import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, D=32):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, D, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(D, 3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(D)


        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        self.apply(init_weights)
    
    def forward(self, img):
        x = self.relu(self.bn1(self.conv1(img)))
        return self.conv2(x) + img
    
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class DeepCNN(nn.Module):
    def __init__(self, D=64, num_blocks=4):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, D, 3, 1, 1),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True)
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