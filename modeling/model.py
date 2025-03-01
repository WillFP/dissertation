import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # 1x1 convolution for shortcut if channel sizes differ, otherwise identity
        self.shortcut = nn.Conv1d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.LeakyReLU(0.01, inplace=True)(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        return nn.LeakyReLU(0.01, inplace=True)(out)


class ChessEvaluator(nn.Module):
    def __init__(self):
        super().__init__()

        self.board_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(12, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=2),
                nn.LeakyReLU(0.2)
            )
        ])
        self.board_flatten = nn.Flatten()

        self.metadata_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),

            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128)
        )

        self.fc_encode = nn.Linear(512 + 128, 128)

        self.features = nn.Sequential(
            ResidualBlock(1, 64),  # Output: (batch_size, 64, 128)
            nn.MaxPool1d(2),  # Output: (batch_size, 64, 64)
            nn.Dropout(0.3),
            ResidualBlock(64, 128),  # Output: (batch_size, 128, 64)
            nn.MaxPool1d(2),  # Output: (batch_size, 128, 32)
            nn.Dropout(0.3),
            ResidualBlock(128, 256),  # Output: (batch_size, 256, 32)
            nn.AdaptiveAvgPool1d(1)  # Output: (batch_size, 256, 1)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),  # Output: (batch_size, 256)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Output: (batch_size, 1)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, board, metadata):
        # Board processing [batch, 12, 8, 8]
        x = board
        for block in self.board_encoder:
            x = block(x)  # [batch, 512, 3, 3]
        board_feats = self.board_flatten(x)  # [batch, 4608]

        # Metadata processing [batch, 5]
        metadata_feats = self.metadata_encoder(metadata)  # [batch, 128]

        # Combine
        combined = torch.cat([board_feats, metadata_feats], dim=1)  # [batch, 4736]
        latent = self.fc_encode(combined)  # [batch, 128]

        # Fix dimensions for residual blocks
        x = latent.unsqueeze(1)  # [batch, 1, 128] instead of [batch, 128, 1]
        features = self.features(x)  # [batch, 256, 1]
        return self.regressor(features)  # [batch, 1]
