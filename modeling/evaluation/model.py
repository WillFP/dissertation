import torch.nn as nn


class ChessEvaluationCNN(nn.Module):
    """
    A 1D CNN for predicting a single evaluation score from a 128-d latent vector.
    The input is (batch_size, 128). We treat that as a 1D "feature map" for convolution.
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            # Final layer for scalar evaluation output
            nn.Linear(20, 1)
        )

    def forward(self, x):
        """
        Forward pass:
          x shape: (batch_size, 128)
        Returns:
          (batch_size, 1) => the predicted evaluation scores.
        """
        # Reshape to (batch_size, 1, 128) for 1D convolution
        out = self.net(x)
        return out
