import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from modeling.autoencoder import ChessAutoencoder
from modeling.dataset import ChessDataset
from modeling.evaluation import ChessEvaluationCNN


def chess_position_loss(predicted, actual):
    """
    Custom loss function for chess position evaluation using pure PyTorch operations.
    - Fixed device compatibility
    - Added numerical stability
    - Adjusted scaling factors
    """
    # Ensure tensors are on same device and cast properly
    predicted = predicted.float()
    actual = actual.float()

    # Sign error penalty (1.0 where signs differ)
    sign_error = (predicted * actual) < 0
    sign_error = sign_error.float()  # Convert to 0/1 tensor

    # Base MSE component with numerical stability
    base_loss = torch.pow(predicted - actual, 2) / 400.0

    # Sigmoid scaling using PyTorch's native implementation
    magnitude_diff = torch.abs(predicted - actual)
    scaled_magnitude = 2.0 * torch.sigmoid(magnitude_diff / 20.0)

    # Combine components with stability epsilon
    total_loss = (base_loss * scaled_magnitude) + (sign_error * 25.0)

    # Mean reduction with non-zero division protection
    return torch.mean(total_loss) + 1e-8


def train_evaluator(
        autoencoder,
        evaluator,
        train_loader,
        val_loader,
        num_epochs=20,
        learning_rate=1e-4,
        device='cpu',
        model_save_path='models/best_eval_model.pt',
        plot_save_path='models/plots/eval_loss.png',
        weight_decay=0.01,
        scheduler_patience=5,
        scheduler_factor=0.1,
        early_stop_patience=0
):
    """
    Train an evaluation model to predict Stockfish eval from (board, metadata).

    We:
      - use the pretrained autoencoder to encode board+metadata -> latent_vec
      - pass latent_vec to evaluator -> predicted_score
      - compute MSE vs actual evaluation

    Args:
        autoencoder: A pretrained ChessAutoencoder (frozen).
        evaluator: The evaluation network to train (EvaluationModel).
        train_loader: DataLoader for training (board, metadata, evaluation).
        val_loader: DataLoader for validation.
        num_epochs: Training epochs.
        learning_rate: AdamW learning rate.
        device: 'cpu', 'cuda', or 'mps'.
        model_save_path: Where to save the best model checkpoint.
        plot_save_path: Where to save the train/val loss plot.
        weight_decay: Weight decay (L2 penalty) for the optimizer.
        scheduler_patience: Patience epochs for LR reduction on plateau (0 disables).
        scheduler_factor: Factor by which LR is reduced.
        early_stop_patience: Early stopping patience epochs (0 disables).

    Returns:
        (train_losses, val_losses)
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

    autoencoder.to(device)
    autoencoder.eval()
    evaluator = evaluator.to(device)

    optimizer = torch.optim.AdamW(evaluator.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize scheduler
    scheduler = None
    if scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=scheduler_patience, factor=scheduler_factor
        )

    # Combine training and val loss, safe with large training sets
    best_combined_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs_no_improve = 0  # For early stopping

    for epoch in range(num_epochs):
        evaluator.train()
        running_train_loss = 0.0
        n_train_batches = 0

        for board, metadata, evaluation in train_loader:
            board = board.to(device)
            metadata = metadata.to(device)
            evaluation = evaluation.to(device).unsqueeze(1)

            with torch.no_grad():
                latent_vec = autoencoder.encode(board, metadata)

            pred_eval = evaluator(latent_vec)
            loss = F.mse_loss(pred_eval, evaluation)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            n_train_batches += 1

        epoch_train_loss = running_train_loss / n_train_batches if n_train_batches > 0 else 0.0
        train_losses.append(epoch_train_loss)

        # Validation phase
        evaluator.eval()
        running_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for board, metadata, evaluation in val_loader:
                board = board.to(device)
                metadata = metadata.to(device)
                evaluation = evaluation.to(device).unsqueeze(1)

                latent_vec = autoencoder.encode(board, metadata)
                pred_eval = evaluator(latent_vec)
                loss = F.mse_loss(pred_eval, evaluation)

                running_val_loss += loss.item()
                n_val_batches += 1

        epoch_val_loss = running_val_loss / n_val_batches if n_val_batches > 0 else 0.0
        val_losses.append(epoch_val_loss)

        # Update learning rate scheduler
        if scheduler:
            scheduler.step(epoch_val_loss)

        epoch_combined_loss = epoch_train_loss + epoch_val_loss

        # Check for improvement
        if epoch_combined_loss < best_combined_loss:
            best_combined_loss = epoch_combined_loss
            epochs_no_improve = 0
            torch.save(evaluator.state_dict(), model_save_path)
            print(f"Epoch {epoch + 1}: new best combined loss = {epoch_combined_loss:.4f}. Model saved.")
        else:
            epochs_no_improve += 1

        # Early stopping check
        if 0 < early_stop_patience <= epochs_no_improve:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs).")
            break

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Combined Loss: {epoch_combined_loss:.4f}"
        )

    # Plotting
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Evaluation Model Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_save_path)
    plt.close()

    print(f"Training complete. Best Combined Loss: {best_combined_loss:.4f}")
    print(f"Loss plot saved to: {plot_save_path}")

    return train_losses, val_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an evaluation model using a pretrained autoencoder')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to HDF5 file with labeled positions (boards, metadata, evaluations)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Latent dimension used by the autoencoder')
    parser.add_argument('--autoencoder-path', type=str, required=True,
                        help='Path to the pretrained autoencoder weights')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to save the model to')
    parser.add_argument('--existing-model', type=str, default=None,
                        help='Path to an existing model to continue training')
    args = parser.parse_args()

    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/plots', exist_ok=True)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataset
    dataset = ChessDataset(args.data)  # (board, metadata, eval)
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

    print("Loading autoencoder...")

    # Load the pretrained autoencoder
    autoencoder = ChessAutoencoder(latent_dim=args.latent_dim)
    autoencoder.load_state_dict(torch.load(args.autoencoder_path, map_location=device, weights_only=True))
    autoencoder.eval()

    print("Loading evaluation model...")
    # Create the evaluation model
    evaluator = ChessEvaluationCNN(latent_dim=args.latent_dim)

    # Load an existing model to continue training
    if args.existing_model:
        evaluator.load_state_dict(torch.load(args.existing_model, map_location=device, weights_only=True))

    print("Training evaluation model...")

    # Train the evaluator
    train_evaluator(
        autoencoder=autoencoder,
        evaluator=evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        model_save_path=args.path
    )
