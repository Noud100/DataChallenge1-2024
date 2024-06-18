import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2,
                                        groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)  # Pointwise convolution

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            ConvNeXtBlock(in_channels, 64, kernel_size=7),  # Adjust `input_channels` here
            nn.ReLU(),
            nn.MaxPool2d(2),
            ConvNeXtBlock(64, 128, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ConvNeXtBlock(128, 256, kernel_size=7),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
