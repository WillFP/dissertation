import argparse

import matplotlib.pyplot as plt

from modeling import ChessDataset


def count_duplicates(dataset, batch_size=10000):
    """
    Count duplicate chess positions in a large ChessDataset efficiently
    using batched processing and hash tables.

    Args:
        dataset: ChessDataset instance
        batch_size: Number of samples to process at once

    Returns:
        tuple: (unique_boards, duplicates)
    """
    from tqdm import tqdm

    seen_boards = {}
    duplicates = 0
    total_samples = len(dataset)

    # Process in batches to avoid memory issues
    for start_idx in tqdm(range(0, total_samples, batch_size)):
        end_idx = min(start_idx + batch_size, total_samples)

        batch_boards = dataset.boards[start_idx:end_idx]

        for i, board in enumerate(batch_boards):
            board_hash = hash(board.cpu().numpy().tobytes())

            if board_hash in seen_boards:
                duplicates += 1
            else:
                seen_boards[board_hash] = 1

    unique_boards = len(seen_boards)

    return unique_boards, duplicates


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a chess evaluation model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to HDF5 file with labeled positions')
    args = parser.parse_args()

    # Load the dataset
    print("Loading dataset...")
    dataset = ChessDataset(args.data)

    # Confirm dataset is loaded with the number of samples
    print(f"Dataset loaded. Checking {len(dataset)} samples.")

    # Count duplicate positions
    unique_boards, duplicates = count_duplicates(dataset)
    print(f"Unique boards: {unique_boards}")
    print(f"Duplicate boards: {duplicates}")
    print(f"Duplicate rate: {(len(dataset) - unique_boards) / len(dataset):.2%}")

    # Print basic statistics for insight into the evaluations
    print(f"Min evaluation: {dataset.evaluations.min().item()}")
    print(f"Max evaluation: {dataset.evaluations.max().item()}")
    print(f"Mean evaluation: {dataset.evaluations.mean().item()}")
    print(f"Standard deviation: {dataset.evaluations.std().item()}")

    # Plot the evaluation distribution
    plt.hist(dataset.evaluations, bins=100)
    plt.xlabel('Evaluation')
    plt.ylabel('Count')
    plt.title('Evaluation Distribution')
    plt.show()
