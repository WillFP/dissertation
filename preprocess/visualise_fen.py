import matplotlib.pyplot as plt
import numpy as np

from preprocess import fen_to_tensor, augment_board


def visualize_reconstruction(board, save_path=None):
    """
    Visualize the original chess position from a tensor representation.

    Args:
        board (np.ndarray): Tensor of shape (8, 8, 12) representing the chess board,
                            with 12 channels for piece types.
        save_path (str, optional): Path to save the visualization.
                                   If None, display on screen.
    """
    plt.figure(figsize=(6, 6))

    def board_to_visual(board_12x8x8):
        visual = np.zeros((8, 8), dtype=np.float64)
        piece_values = {
            0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6,  # White: P, N, B, R, Q, K
            6: -1, 7: -2, 8: -3, 9: -4, 10: -5, 11: -6  # Black: P, N, B, R, Q, K
        }
        for i, val in piece_values.items():
            visual += board_12x8x8[:, :, i].astype(np.int32) * val
        return visual

    visual = board_to_visual(board)

    plt.imshow(visual, cmap='RdBu', vmin=-6, vmax=6)
    #lt.title('Encoded Position')
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # Convert single tensor to batch format (4D)
    tensor, metadata = fen_to_tensor("2q5/3B1Pk1/1K6/P3P3/p4Rb1/4nP1P/1pp5/6N1 w - - 0 1")
    visualize_reconstruction(tensor, "encoded.png")

    # Add batch dimension to make it 4D for NumPy array
    batch_tensor = np.expand_dims(tensor, axis=0)  # Adds dimension at position 0

    # Augment the batched tensor
    augmented_batch = augment_board(batch_tensor)

    # Extract the first (and only) augmented tensor
    augmented_tensor = augmented_batch[0]

    # Visualize the augmented tensor
    visualize_reconstruction(augmented_tensor, "augmented.png")
