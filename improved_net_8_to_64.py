import torch
import torch.nn as nn

# activation_func = nn.Tanhshrink()
activation_func = nn.LeakyReLU()
# overfitting data augmnentations / dropout

class ImprovedNet_8_64(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(ImprovedNet_8_64, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),  # make it your out_channels
            activation_func,

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),  # make it your out_channels
            activation_func,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # make it your out_channels
            activation_func,

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # make it your out_channels
            activation_func,
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 6, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(6),  # make it your out_channels
            activation_func,
        )

        # Calculate the flattened size after the convolutional layers
        self.flatten_size = self._get_flatten_size((1, 128, 128))

        self.linear_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            activation_func,
            nn.Linear(256, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def _get_flatten_size(self, input_size):
        x = torch.randn(1, *input_size)
        x = self.cnn_layers(x)
        return x.view(1, -1).size(1)
