import argparse
import os
import time
from functools import partial
import hashlib

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data._utils.collate import default_collate

from modeling.autoencoder import ChessAutoencoder
from modeling.dataset import ChessDataset
from modeling.evaluation import ChessEvaluationCNN


def chess_position_loss(predicted, actual):
    """
    Custom loss function for chess position evaluation using pure PyTorch operations.

    Args:
        predicted (torch.Tensor): Predicted evaluation scores
        actual (torch.Tensor): Actual evaluation scores

    Returns:
        torch.Tensor: Mean loss value with added numerical stability
    """
    predicted = predicted.float()
    actual = actual.float()

    # Sign error penalty
    sign_error = (predicted * actual) < 0
    sign_error = sign_error.float()

    # Base squared error loss, scaled down
    base_loss = torch.pow(predicted - actual, 2) / 400.0

    # Magnitude difference with sigmoid scaling
    magnitude_diff = torch.abs(predicted - actual)
    scaled_magnitude = 2.0 * torch.sigmoid(magnitude_diff / 20.0)

    # Combined loss with sign penalty
    total_loss = (base_loss * scaled_magnitude) + (sign_error * 25.0)

    return torch.mean(total_loss) + 1e-8


def create_latent_dataset(dataset, autoencoder, batch_size, device, num_workers=8, cache_file=None):
    """
    Precompute latent vectors from the dataset using the autoencoder with caching.

    Args:
        dataset (ChessDataset): Input dataset
        autoencoder (ChessAutoencoder): Pretrained autoencoder model
        batch_size (int): Batch size for encoding
        device (str): Device to run encoding on ('cuda', 'mps', or 'cpu')
        num_workers (int): Number of data loader workers
        cache_file (str, optional): Path to cache file for saving/loading latents

    Returns:
        TensorDataset: Dataset of latent vectors and evaluations
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading precomputed latent vectors from cache: {cache_file}")
        try:
            cached_data = torch.load(cache_file)
            print(f"Cache loaded successfully with {len(cached_data[0])} samples")
            return TensorDataset(*cached_data)
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    print("Precomputing latent vectors with accelerated setup...")
    start_time = time.time()

    effective_batch_size = min(batch_size * 4, 2048)

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
    )

    print(f"Using batch size {effective_batch_size} for encoding ({len(dataset)} total samples)")

    total_batches = len(loader)
    latent_vecs = []
    evaluations = []

    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    print(f"Autoencoder on device: {next(autoencoder.parameters()).device}")

    processed_samples = 0
    last_time = time.time()

    use_amp = device == 'cuda'

    if use_amp:
        print("Using mixed precision for faster encoding")
        context = torch.amp.autocast('cuda')
    else:
        from contextlib import nullcontext
        context = nullcontext()

    with torch.no_grad():
        for batch_idx, (board, metadata, evaluation) in enumerate(loader):
            board = board.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)

            if batch_idx == 0:
                print(f"Input board device: {board.device}")

            with context:
                try:
                    latent_vec = autoencoder.encode(board, metadata)
                except RuntimeError as e:
                    print(f"Error encoding batch {batch_idx}: {e}")
                    if 'expected device' in str(e).lower():
                        print("Device mismatch. Fixing and retrying...")
                        autoencoder = autoencoder.to(board.device)
                        latent_vec = autoencoder.encode(board, metadata)
                    else:
                        raise

            # Cast to float32 before moving to CPU to match model precision
            latent_vecs.append(latent_vec.float().cpu())
            evaluations.append(evaluation)

            processed_samples += board.size(0)
            current_time = time.time()

            if batch_idx % 10 == 0 or (current_time - last_time) > 5:
                elapsed = current_time - start_time
                samples_per_sec = processed_samples / elapsed
                pct_complete = 100 * (batch_idx + 1) / total_batches
                eta = (total_batches - (batch_idx + 1)) * elapsed / (batch_idx + 1)

                print(f"Processed {batch_idx + 1}/{total_batches} batches "
                      f"({pct_complete:.1f}%) | "
                      f"{samples_per_sec:.1f} samples/sec | "
                      f"ETA: {eta / 60:.1f} min")

                last_time = current_time

            if device == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

    print("Concatenating results...")
    latent_vecs = torch.cat(latent_vecs, dim=0)
    evaluations = torch.cat(evaluations, dim=0).flatten()

    total_time = time.time() - start_time
    print(f"Completed encoding {len(latent_vecs)} samples in {total_time:.1f}s "
          f"({len(latent_vecs) / total_time:.1f} samples/sec)")

    if cache_file:
        print(f"Saving latent vectors to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save((latent_vecs, evaluations), cache_file)

    return TensorDataset(latent_vecs, evaluations)


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
        autoencoder=None,
        evaluator=None,
        train_loader=None,
        val_loader=None,
        num_epochs=20,
        learning_rate=1e-4,
        device='cpu',
        model_save_path='models/best_eval_model.pt',
        plot_save_path='models/plots/eval_loss.png',
        weight_decay=0.01,
        scheduler_patience=5,
        scheduler_factor=0.1,
        early_stop_patience=0,
        precomputed_dataset=False
):
    """
    Train the evaluation model using either precomputed latents or on-the-fly encoding.

    Args:
        autoencoder (ChessAutoencoder, optional): Pretrained autoencoder
        evaluator (ChessEvaluationCNN): Evaluation model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        device (str): Device to use
        model_save_path (str): Path to save best model
        plot_save_path (str): Path to save loss plot
        weight_decay (float): Weight decay for optimizer
        scheduler_patience (int): Patience for LR scheduler
        scheduler_factor (float): Factor to reduce LR
        early_stop_patience (int): Patience for early stopping
        precomputed_dataset (bool): Whether latents are precomputed
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

    evaluator = evaluator.to(device)
    if not precomputed_dataset and autoencoder is not None:
        autoencoder = autoencoder.to(device)
        autoencoder.eval()

    optimizer = torch.optim.Adam(
        evaluator.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=scheduler_patience,
        factor=scheduler_factor, verbose=True
    )

    best_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        evaluator.train()
        running_train_loss = 0.0

        train_iter = train_loader

        for batch in train_iter:
            optimizer.zero_grad()
            if precomputed_dataset:
                latent_vec, evaluation = batch
                latent_vec = latent_vec.to(device)
                evaluation = evaluation.to(device).unsqueeze(1)
            else:
                board, metadata, evaluation = batch
                board = board.to(device)
                metadata = metadata.to(device)
                evaluation = evaluation.to(device).unsqueeze(1)
                autoencoder = autoencoder.to(device)
                with torch.no_grad():
                    latent_vec = autoencoder.encode(board, metadata)

            pred_eval = evaluator(latent_vec)
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
                if precomputed_dataset:
                    latent_vec, evaluation = batch
                    latent_vec = latent_vec.to(device)
                    evaluation = evaluation.to(device).unsqueeze(1)
                else:
                    board, metadata, evaluation = batch
                    board = board.to(device)
                    metadata = metadata.to(device)
                    evaluation = evaluation.to(device).unsqueeze(1)
                    latent_vec = autoencoder.encode(board, metadata)

                pred_eval = evaluator(latent_vec)
                loss = chess_position_loss(pred_eval, evaluation)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)
        epoch_time = time.time() - epoch_start_time

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(evaluator.state_dict(), model_save_path)
            print(f"Epoch {epoch + 1}: new best val loss = {epoch_val_loss:.4f}. Model saved.")
        else:
            epochs_no_improve += 1

        if 0 < early_stop_patience <= epochs_no_improve:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {early_stop_patience} epochs).")
            break

        print(
            f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s | "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if device == 'cuda':
            torch.cuda.empty_cache()

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

    print(f"Training complete. Best val loss: {best_loss:.4f}")
    return train_losses, val_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an evaluation model using a pretrained autoencoder')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to HDF5 file with labeled positions (boards, metadata, evaluations)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--encode-batch-size', type=int, default=None,
                        help='Batch size for encoding (defaults to min(batch_size * 4, 2048))')
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
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate for the optimizer')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--precompute-latents', action='store_true',
                        help='Precompute all latent vectors (saves computation during training)')
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

    print("Loading autoencoder...")
    autoencoder = ChessAutoencoder(latent_dim=args.latent_dim)
    autoencoder.load_state_dict(torch.load(args.autoencoder_path, map_location=device, weights_only=True))
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    print(f"Verifying autoencoder device: {next(autoencoder.parameters()).device}")

    print("Loading dataset...")
    dataset = ChessDataset(args.data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    precomputed_dataset = False
    if args.precompute_latents:
        print("Precomputing latent vectors for faster training...")
        os.makedirs("cache", exist_ok=True)

        autoencoder_hash = hashlib.md5(
            torch.cat([p.flatten() for p in autoencoder.parameters()]).cpu().detach().numpy().tobytes()
        ).hexdigest()[:8]
        train_cache_file = f"cache/latent_train_{args.latent_dim}_{autoencoder_hash}.pt"
        val_cache_file = f"cache/latent_val_{args.latent_dim}_{autoencoder_hash}.pt"

        encode_batch_size = args.encode_batch_size if args.encode_batch_size else min(args.batch_size * 4, 2048)
        train_latent_dataset = create_latent_dataset(train_dataset, autoencoder, encode_batch_size, device,
                                                     cache_file=train_cache_file)
        val_latent_dataset = create_latent_dataset(val_dataset, autoencoder, encode_batch_size, device,
                                                   cache_file=val_cache_file)
        precomputed_dataset = True

        train_loader = DataLoader(
            train_latent_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device != 'cpu',
        )
        val_loader = DataLoader(
            val_latent_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device != 'cpu',
        )
    else:
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
    evaluator = ChessEvaluationCNN(latent_dim=args.latent_dim)
    if args.existing_model:
        evaluator.load_state_dict(torch.load(args.existing_model, map_location=device, weights_only=True))

    print("Training evaluation model...")
    train_evaluator(
        autoencoder=autoencoder,
        evaluator=evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        model_save_path=args.path,
        precomputed_dataset=precomputed_dataset
    )
