import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from modeling.autoencoder import ChessAutoencoder
from modeling.dataset import ChessDataset


def train_autoencoder(
        model,
        train_loader,
        val_loader,
        num_epochs=20,
        learning_rate=1e-4,
        device='cpu',
        model_save_path='models/best_autoencoder.pt',
        plot_save_path='models/plots/autoencoder_loss.png'
):
    """
    Train a pure autoencoder (no KL). We'll do:
      - board reconstruction: BCE
      - metadata reconstruction: MSE
      - Weighted sum: total_loss = board_bce + 0.1 * metadata_mse
    We'll track train/val loss each epoch, plot them, and save best model.

    Args:
        model: A ChessAutoencoder instance.
        train_loader: Dataloader for training data.
        val_loader: Dataloader for validation data.
        num_epochs: Number of epochs to train.
        learning_rate: Adam learning rate.
        device: 'cpu', 'cuda', or 'mps'.
        model_save_path: Where to save the best model.
        plot_save_path: Where to save the train/val loss plot.

    Returns:
        (train_losses, val_losses, train_board_losses, val_board_losses, train_meta_losses, val_meta_losses)
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    train_board_losses = []
    val_board_losses = []
    train_meta_losses = []
    val_meta_losses = []

    def compute_loss(board_out, board, meta_out, metadata):
        """
        BCE for board, MSE for metadata, final = BCE + 0.1*MSE
        """
        bce = F.binary_cross_entropy(
            board_out.view(-1),
            board.view(-1),
            reduction='mean'
        )
        mse = F.mse_loss(meta_out, metadata, reduction='mean')
        total = bce + 0.1 * mse
        return total, bce, mse

    for epoch in range(num_epochs):
        # ---------------------------
        #         TRAIN PHASE
        # ---------------------------
        model.train()

        running_train_loss = 0.0
        running_train_bce = 0.0
        running_train_mse = 0.0
        n_train_batches = 0

        for board, metadata in train_loader:
            board = board.to(device)
            metadata = metadata.to(device)

            board_out, meta_out = model(board, metadata)
            loss, bce, mse = compute_loss(board_out, board, meta_out, metadata)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            running_train_bce += bce.item()
            running_train_mse += mse.item()
            n_train_batches += 1

        if n_train_batches > 0:
            epoch_train_loss = running_train_loss / n_train_batches
            epoch_train_bce = running_train_bce / n_train_batches
            epoch_train_mse = running_train_mse / n_train_batches
        else:
            epoch_train_loss, epoch_train_bce, epoch_train_mse = 0.0, 0.0, 0.0

        train_losses.append(epoch_train_loss)
        train_board_losses.append(epoch_train_bce)
        train_meta_losses.append(epoch_train_mse)

        # ---------------------------
        #       VALIDATION PHASE
        # ---------------------------
        model.eval()

        running_val_loss = 0.0
        running_val_bce = 0.0
        running_val_mse = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for board, metadata in val_loader:
                board = board.to(device)
                metadata = metadata.to(device)

                board_out, meta_out = model(board, metadata)
                loss, bce, mse = compute_loss(board_out, board, meta_out, metadata)

                running_val_loss += loss.item()
                running_val_bce += bce.item()
                running_val_mse += mse.item()
                n_val_batches += 1

        if n_val_batches > 0:
            epoch_val_loss = running_val_loss / n_val_batches
            epoch_val_bce = running_val_bce / n_val_batches
            epoch_val_mse = running_val_mse / n_val_batches
        else:
            epoch_val_loss, epoch_val_bce, epoch_val_mse = 0.0, 0.0, 0.0

        val_losses.append(epoch_val_loss)
        val_board_losses.append(epoch_val_bce)
        val_meta_losses.append(epoch_val_mse)

        # ---------------------------
        #     MODEL SAVING
        # ---------------------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch {epoch + 1}: new best val loss = {best_val_loss:.4f}. Model saved.")

        # Print summary
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
            f"(Board BCE: {epoch_val_bce:.4f}, Metadata MSE: {epoch_val_mse:.4f})"
        )

    # ---------------------------
    #   PLOTTING
    # ---------------------------
    import matplotlib.pyplot as plt

    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Autoencoder Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_save_path)
    plt.close()

    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"Loss plot saved to: {plot_save_path}")

    return (
        train_losses, val_losses,
        train_board_losses, val_board_losses,
        train_meta_losses, val_meta_losses
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train chess position autoencoder')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to HDF5 file with labeled positions')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Dimension of latent space')
    args = parser.parse_args()

    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/plots', exist_ok=True)

    # Automatically determine device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataset
    dataset = ChessDataset(args.data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create model
    model = ChessAutoencoder(latent_dim=args.latent_dim)

    train_autoencoder(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        device=device
    )
