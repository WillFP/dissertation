from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from modeling.autoencoder import ChessAutoencoder


def load_model(model_path, latent_dim=64, device='cpu'):
    """
    Load a trained (pure) autoencoder model from disk.

    Args:
        model_path (str): Path to the saved model checkpoint
        latent_dim (int): Dimension of the latent space
        device (str): Device to load the model on ('cpu' or 'cuda')

    Returns:
        model: Loaded ChessAutoencoder model
    """
    model = ChessAutoencoder(latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def encode_position(model, board, metadata, device='cpu'):
    """
    Encode a single chess position into the latent space.

    Args:
        model: Trained ChessAutoencoder model (pure AE).
        board (torch.Tensor): Board tensor of shape (12, 8, 8)
        metadata (torch.Tensor): Metadata tensor of shape (5,)
        device (str): Device to perform computation on

    Returns:
        latent: Latent space representation (a single vector)
    """
    model.eval()
    with torch.no_grad():
        board = board.unsqueeze(0).to(device)  # Add batch dimension: (1, 12, 8, 8)
        metadata = metadata.unsqueeze(0).to(device)  # (1, 5)
        latent = model.encode(board, metadata)  # shape (1, latent_dim)
    return latent.cpu().numpy()


def decode_position(model, latent, device='cpu'):
    """
    Decode a latent representation back to a chess position.

    Args:
        model: Trained ChessAutoencoder model
        latent (np.ndarray): Latent space representation, shape (latent_dim,) or (batch, latent_dim)
        device (str): Device to perform computation on

    Returns:
        (reconstructed_board, reconstructed_metadata)
        - reconstructed_board: shape (batch, 12, 8, 8) in [0,1]
        - reconstructed_metadata: shape (batch, 5) in [0,1]
    """
    model.eval()
    with torch.no_grad():
        # If latent is a single vector, expand to (1, latent_dim).
        # If it's already (batch_size, latent_dim), let it be.
        if len(latent.shape) == 1:
            latent = latent[np.newaxis, :]  # shape becomes (1, latent_dim)

        latent_tensor = torch.FloatTensor(latent).to(device)
        board_out, metadata_out = model.decode(latent_tensor)
    return board_out.cpu().numpy(), metadata_out.cpu().numpy()


def visualize_reconstruction(original_board, reconstructed_board, save_path=None):
    """
    Visualize the original and reconstructed chess positions side by side.

    Args:
        original_board (np.ndarray): shape (12, 8, 8)
        reconstructed_board (np.ndarray): shape (12, 8, 8)
        save_path (str, optional): Path to save the visualization.
                                   If None, display on screen.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Convert 12-channel representation to a single channel for visualization
    def board_to_visual(board_12x8x8):
        visual = np.zeros((8, 8))
        piece_values = {
            0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6,
            6: -1, 7: -2, 8: -3, 9: -4, 10: -5, 11: -6
        }
        for i, val in piece_values.items():
            visual += board_12x8x8[i] * val
        return visual

    original_visual = board_to_visual(original_board)
    reconstructed_visual = board_to_visual(reconstructed_board)

    ax1.imshow(original_visual, cmap='RdBu', vmin=-6, vmax=6)
    ax1.set_title('Original Position')
    ax1.axis('off')

    ax2.imshow(reconstructed_visual, cmap='RdBu', vmin=-6, vmax=6)
    ax2.set_title('Reconstructed Position')
    ax2.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='Chess position autoencoder predictions')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to HDF5 file with positions to encode')
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='Directory to save predictions and visualizations')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Dimension of latent space')
    parser.add_argument('--num-examples', type=int, default=5,
                        help='Number of example reconstructions to visualize')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends,
                                                                       'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model, args.latent_dim, device)

    # Load dataset
    # Make sure this matches your local structure: 'modeling.dataset' vs 'modeling'
    from modeling.dataset import ChessDataset

    dataset = ChessDataset(args.data)

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # adjust if needed
        pin_memory=(device == 'cuda')  # pin_memory is beneficial if using GPU
    )

    # Encode all positions in the dataset
    all_latents = []
    print("Encoding positions...")

    with torch.no_grad():
        for i, (boards, metadata, _) in enumerate(dataloader):
            boards = boards.to(device)
            metadata = metadata.to(device)

            # Pure AE: 'encode' returns just the latent vector z
            latent_z = model.encode(boards, metadata)  # shape (batch_size, latent_dim)
            all_latents.append(latent_z.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"Processed {(i + 1) * args.batch_size} positions...")

    all_latents = np.concatenate(all_latents, axis=0)

    # Save latent representations
    latent_file = output_dir / 'latent_representations.npy'
    np.save(latent_file, all_latents)
    print(f"Saved latent representations to {latent_file}")

    # Visualize some random examples
    print("Generating example reconstructions...")
    indices = np.random.choice(len(dataset), args.num_examples, replace=False)

    for i, idx in enumerate(indices):
        board, metadata, _ = dataset[idx]
        # Encode single example
        latent_vec = encode_position(model, board, metadata, device=device)  # shape (1, latent_dim) in np
        # Decode back
        reconstructed_board, reconstructed_metadata = decode_position(model, latent_vec, device=device)
        # reconstructed_board shape => (1, 12, 8, 8), we want the first batch
        reconstructed_board = reconstructed_board[0]

        # Visualize
        out_file = output_dir / f'reconstruction_{i}.png'
        visualize_reconstruction(
            original_board=board.numpy(),
            reconstructed_board=reconstructed_board,
            save_path=out_file
        )

    print(f"Saved {args.num_examples} example reconstructions in {output_dir}")
