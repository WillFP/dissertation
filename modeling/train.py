import argparse
import os
import time
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data._utils.collate import default_collate

from modeling.dataset import ChessDataset
from modeling.model import ChessEvaluator


def chess_position_loss(predicted, actual, delta=5.0, sign_penalty=25.0):
    """
    Improved loss function for chess position evaluation using PyTorch.

    Args:
        predicted (torch.Tensor): Predicted evaluation scores
        actual (torch.Tensor): Actual evaluation scores
        delta (float): Delta parameter for Huber loss (default: 1.0)
        sign_penalty (float): Penalty for sign errors (default: 25.0)

    Returns:
        torch.Tensor: Mean loss value
    """
    predicted = predicted.float()
    actual = actual.float()

    # Huber loss
    huber_loss = nn.HuberLoss(reduction='none', delta=delta)(predicted, actual)

    # Sign error penalty
    sign_error = (predicted * actual) < 0
    sign_error = sign_error.float()

    # Total loss
    total_loss = huber_loss + sign_error * sign_penalty

    return torch.mean(total_loss)


def gpu_collate(batch, device):
    """
    Move tensors to GPU during collation to minimize CPU-GPU transfer time.

    Args:
        batch: Batch of data
        device (str): Target device

    Returns:
        list: Collated batch moved to device
    """
    batch = default_collate(batch)
    return [t.to(device, non_blocking=True) for t in batch]


def train_evaluator(
        evaluator=None,
        train_loader=None,
        val_loader=None,
        num_epochs=20,
        learning_rate=1e-4,
        min_learning_rate=1e-7,
        device='cpu',
        model_save_path='models/best_eval_model.pt',
        plot_save_path='models/plots/eval_loss.png',
        weight_decay=0.0001,
):
    """
    Train the evaluation model using either precomputed latents or on-the-fly encoding.

    Args:
        evaluator (ChessEvaluationCNN): Evaluation model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        device (str): Device to use
        model_save_path (str): Path to save best model
        plot_save_path (str): Path to save loss plot
        weight_decay (float): Weight decay for optimizer
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

    evaluator = evaluator.to(device)

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

    best_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs_no_improve = 0

    def save_model():
        torch.save(evaluator.state_dict(), model_save_path)
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Evaluation Model Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_save_path)
        plt.close()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        evaluator.train()
        running_train_loss = 0.0

        train_iter = train_loader

        for batch in train_iter:
            optimizer.zero_grad()
            board, metadata, evaluation = batch
            board = board.to(device)
            metadata = metadata.to(device)
            evaluation = evaluation.to(device).unsqueeze(1)
            pred_eval = evaluator(board, metadata)
            loss = chess_position_loss(pred_eval, evaluation)

            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        evaluator.eval()
        running_val_loss = 0.0
        val_iter = val_loader

        with torch.no_grad():
            for batch in val_iter:
                board, metadata, evaluation = batch
                board = board.to(device)
                metadata = metadata.to(device)
                evaluation = evaluation.to(device).unsqueeze(1)
                pred_eval = evaluator(board, metadata)
                loss = chess_position_loss(pred_eval, evaluation)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)
        epoch_time = time.time() - epoch_start_time

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            epochs_no_improve = 0
            save_model()
            print(f"Epoch {epoch + 1}: new best val loss = {epoch_val_loss:.4f}.")
        else:
            epochs_no_improve += 1

        if epoch % 20 == 0:
            save_model()

        print(
            f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s | "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if device == 'cuda':
            torch.cuda.empty_cache()

    save_model()
    print(f"Training complete. Best val loss: {best_loss:.4f}")
    print(f"Saved final model to: {model_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an evaluation model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to HDF5 file with labeled positions (boards, metadata, evaluations)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to save the model to')
    parser.add_argument('--existing-model', type=str, default=None,
                        help='Path to an existing model to continue training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer')
    parser.add_argument('--min-learning-rate', type=float, default=1e-7,
                        help='Minimum learning rate for the optimizer')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    os.makedirs('models/plots', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            current_stream = torch.cuda.current_stream()
            if hasattr(torch.cuda, "StreamPriority") and hasattr(current_stream, "priority"):
                current_stream.priority = torch.cuda.StreamPriority.HIGH
        except (AttributeError, RuntimeError) as e:
            print(f"Note: Could not set CUDA stream priority: {e}")

    print("Loading dataset...")
    dataset = ChessDataset(args.data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if device != 'cpu':
        collate_fn = partial(gpu_collate, device=device)
    else:
        collate_fn = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device != 'cpu' and collate_fn is None,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device != 'cpu' and collate_fn is None,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Dataset loaded. Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples.")
    print(f"Using batch size: {args.batch_size}, workers: {args.num_workers}")

    print("Loading evaluation model...")
    evaluator = ChessEvaluator()
    if args.existing_model:
        evaluator.load_state_dict(torch.load(args.existing_model, map_location=device, weights_only=True))

    print("Training evaluation model...")
    train_evaluator(
        evaluator=evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        device=device,
        model_save_path=args.path,
    )
