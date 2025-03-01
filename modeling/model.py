import torch
import torch.nn as nn


class ChessEvaluator(nn.Module):
    def __init__(self):
        super().__init__()

        self.board_encoder = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, kernel_size=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.2),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        self.metadata_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128)
        )

        self.fc_encode = nn.Linear(1024 + 128, 128)

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
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
        board_feats = self.board_encoder(board)
        metadata_feats = self.metadata_encoder(metadata)

        combined = torch.cat([board_feats, metadata_feats], dim=1)
        latent = self.fc_encode(combined)

        return self.regressor(latent)
