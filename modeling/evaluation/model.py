import torch.nn as nn


class ChessEvaluationCNN(nn.Module):
    """
    A 1D CNN for predicting a single evaluation score from a 128-d latent vector.
    The input is (batch_size, 128). We treat that as a 1D "feature map" for convolution.
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # Reshape from (B, 128) -> (B, 1, 128),
            # then convolve over the "128" dimension with 1 channel in.

            # Conv1: 1->32 channels, kernel_size=3
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            # Conv2: 32->64 channels, kernel_size=3
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),

            # Global average pool over the spatial dimension (i.e., reduce from length=128 to length=1)
            nn.AdaptiveAvgPool1d(output_size=1),

            # Flatten from (B, 64, 1) to (B, 64)
            nn.Flatten(),

            # Finally, a linear layer for the single scalar evaluation output
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Forward pass:
          x shape: (batch_size, 128)
        Returns:
          (batch_size, 1) => the predicted evaluation scores.
        """
        # Reshape to (batch_size, 1, 128) for 1D convolution
        x = x.unsqueeze(1)  # Add a channel dimension
        out = self.net(x)
        return out
