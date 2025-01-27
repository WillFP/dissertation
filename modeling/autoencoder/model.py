import torch
import torch.nn as nn


class ChessAutoencoder(nn.Module):
    def __init__(self, latent_dim=64, dropout_rate=0.1):
        """
        A pure autoencoder:
          - Board encoder: 12x8x8 -> 512 features
          - Metadata encoder: 5 -> 128 features
          - Combined -> latent_dim
          - Then decode back to board + metadata
        """
        super().__init__()

        # ------------------
        # Board Encoder
        # ------------------
        self.board_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(12, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(dropout_rate)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(dropout_rate)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(dropout_rate)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=2),
                nn.LeakyReLU(0.2)
                # final layer: no BN or dropout
            )
        ])
        self.board_flatten = nn.Flatten()

        # ------------------
        # Metadata Encoder
        # ------------------
        self.metadata_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128)
            # no dropout at end
        )

        # ------------------
        # Latent "bottleneck"
        # ------------------
        self.fc_encode = nn.Linear(512 + 128, latent_dim)

        # ------------------
        # Decoder from latent
        # ------------------
        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 + 128),
            nn.LeakyReLU(0.2)
            # keep it simple
        )

        # ------------------
        # Board Decoder
        # ------------------
        self.board_decoder = nn.ModuleList([
            # Step 1: project from 512 -> 512, then reshape to (512,1,1)
            nn.Sequential(
                nn.Linear(512, 512),
                nn.LeakyReLU(0.2)
                # no BN or dropout
            ),
            # Step 2: deconv pipeline
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=2),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(dropout_rate)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(dropout_rate)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2)
                # final BN/Dropout removed
            )
        ])
        self.board_output = nn.Sequential(
            nn.Conv2d(64, 12, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # ------------------
        # Metadata Decoder
        # ------------------
        self.metadata_decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 5),
            nn.Sigmoid()
        )

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def encode(self, board, metadata):
        # Encode board -> 512
        x = board
        for block in self.board_encoder:
            x = block(x)
        board_feats = self.board_flatten(x)

        # Encode metadata -> 128
        meta_feats = self.metadata_encoder(metadata)

        # Combine
        combined = torch.cat([board_feats, meta_feats], dim=1)

        # Go to latent
        z = self.fc_encode(combined)
        return z

    def decode(self, z):
        # Expand back into board vs. metadata features
        expanded = self.latent_decoder(z)

        board_feats = expanded[:, :512]
        meta_feats = expanded[:, 512:]

        # Board decode
        x = self.board_decoder[0](board_feats)
        x = x.view(-1, 512, 1, 1)

        for block in self.board_decoder[1:]:
            x = block(x)
        board_out = self.board_output(x)

        # Metadata decode
        metadata_out = self.metadata_decoder(meta_feats)

        return board_out, metadata_out

    def forward(self, board, metadata):
        z = self.encode(board, metadata)
        board_recon, metadata_recon = self.decode(z)
        return board_recon, metadata_recon
