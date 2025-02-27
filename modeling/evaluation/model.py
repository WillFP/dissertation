import torch.nn as nn

class ChessEvaluationCNN(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128, 256),  # Adjust based on output size without pooling
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, 128)
        features = self.features(x)
        return self.regressor(features)
