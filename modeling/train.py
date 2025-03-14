import argparse
import multiprocessing
import os
import time

import matplotlib.pyplot as plt
import torch
from modeling import ChessEvaluator, ChessDataset
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def chess_position_loss(predicted, actual, delta=5.0, sign_penalty=5.0):
    """
    Improved loss function for chess position evaluation using PyTorch.
    """
    predicted = predicted.float()
    actual = actual.float()
    huber_loss = nn.HuberLoss(reduction='none', delta=delta)(predicted, actual)
    sign_error = (predicted * actual) < 0
    sign_error = sign_error.float()
    total_loss = huber_loss + sign_error * sign_penalty
    return torch.mean(total_loss)


def gpu_collate(batch, device):
    """
    Move tensors to GPU during collation to minimize CPU-GPU transfer time.
    """
    batch = default_collate(batch)
    return [t.to(device, non_blocking=True) for t in batch]


def train_evaluator(
        evaluator,
        train_loader,
        val_loader,
        num_epochs=20,
        learning_rate=1e-4,
        device='cpu',
        model_save_path='models/best_eval_model.pt',
        plot_save_path='models/plots/eval_loss.png',
        weight_decay=0.0001,
):
    """
    Train the chess evaluation model with optimization and logging.

    Args:
        evaluator (nn.Module): The ChessEvaluator model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Initial learning rate.
        device (str): Device to train on ('cpu', 'cuda', or 'mps').
        model_save_path (str): Path to save the best model.
        plot_save_path (str): Path to save loss plot.
        weight_decay (float): Weight decay for AdamW optimizer.
    """
    # Ensure directories exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

    # Move model to device
    evaluator = evaluator.to(device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        evaluator.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1e4,
        anneal_strategy='cos'
    )

    # Initialize AMP scaler for CUDA
    scaler = GradScaler() if device == 'cuda' else None

    # Tracking variables
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    def save_model(graphs_only=False):
        """Save the model state and plot losses."""
        if not graphs_only:
            torch.save(evaluator.state_dict(), model_save_path)
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_save_path)
        plt.close()

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        evaluator.train()
        running_train_loss = 0.0

        # Training progress bar
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")
        for batch in train_iter:
            optimizer.zero_grad()
            board, metadata, evaluation = batch
            # Move tensors to device
            board = board.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            evaluation = evaluation.to(device, non_blocking=True).unsqueeze(1)

            # Forward and backward pass
            if scaler:  # Use AMP on CUDA
                with autocast("cuda"):
                    pred_eval = evaluator(board, metadata)
                    loss = chess_position_loss(pred_eval, evaluation)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(evaluator.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_eval = evaluator(board, metadata)
                loss = chess_position_loss(pred_eval, evaluation)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(evaluator.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            running_train_loss += loss.item()
            train_iter.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation phase
        evaluator.eval()
        running_val_loss = 0.0
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation")
        with torch.no_grad():
            for batch in val_iter:
                board, metadata, evaluation = batch
                board = board.to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)
                evaluation = evaluation.to(device, non_blocking=True).unsqueeze(1)
                if device == 'cuda':
                    with autocast("cuda"):
                        pred_eval = evaluator(board, metadata)
                        loss = chess_position_loss(pred_eval, evaluation)
                else:
                    pred_eval = evaluator(board, metadata)
                    loss = chess_position_loss(pred_eval, evaluation)
                running_val_loss += loss.item()
                val_iter.set_postfix(loss=loss.item())

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        epoch_time = time.time() - epoch_start_time

        # Save model if improved or every 20 epochs
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            save_model()
            print(f"Epoch {epoch + 1}: New best val loss = {epoch_val_loss:.4f}")
        elif (epoch + 1) % 20 == 0:
            save_model(graphs_only=True)

        print(
            f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s | "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if device == 'cuda':
            torch.cuda.empty_cache()

    # Final save
    save_model()
    print(f"Training complete. Best val loss: {best_loss:.4f}")
    print(f"Saved final model to: {model_save_path}")


if __name__ == '__main__':
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a chess evaluation model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to HDF5 file with labeled positions')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to save the model')
    parser.add_argument('--existing-model', type=str, default=None,
                        help='Path to an existing model to continue training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    args = parser.parse_args()

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/plots', exist_ok=True)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Optimize CUDA settings
    if device == 'cuda':
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Load dataset
    print("Loading dataset...")
    dataset = ChessDataset(args.data)
    # 90:10 test/train split because data sets are large and expensive to create
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Configure DataLoaders
    pin_memory = device != 'cpu'
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Dataset loaded. Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples.")
    print(f"Using batch size: {args.batch_size}, workers: {args.num_workers}")

    # Initialize model
    print("Loading evaluation model...")
    evaluator = ChessEvaluator()
    if args.existing_model:
        evaluator.load_state_dict(torch.load(args.existing_model, map_location=device, weights_only=True))

    # Train the model
    print("Training evaluation model...")
    train_evaluator(
        evaluator=evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        model_save_path=args.path,
    )
