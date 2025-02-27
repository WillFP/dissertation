import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # 1x1 convolution for shortcut if channel sizes differ, otherwise identity
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.LeakyReLU(0.01, inplace=True)(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        return nn.LeakyReLU(0.01, inplace=True)(out)

class ChessEvaluationCNN(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            ResidualBlock(1, 64),    # Output: (batch_size, 64, 128)
            nn.MaxPool1d(2),         # Output: (batch_size, 64, 64)
            ResidualBlock(64, 128),  # Output: (batch_size, 128, 64)
            nn.MaxPool1d(2),         # Output: (batch_size, 128, 32)
            ResidualBlock(128, 256), # Output: (batch_size, 256, 32)
            nn.AdaptiveAvgPool1d(1)  # Output: (batch_size, 256, 1)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),            # Output: (batch_size, 256)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)         # Output: (batch_size, 1)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim: (batch_size, 1, 128)
        features = self.features(x)
        return self.regressor(features)
