import argparse
import matplotlib.pyplot as plt
from modeling import ChessDataset

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

    # Print basic statistics for insight into the evaluations
    print(f"Min evaluation: {dataset.evaluations.min().item()}")
    print(f"Max evaluation: {dataset.evaluations.max().item()}")
    print(f"Mean evaluation: {dataset.evaluations.mean().item()}")
    print(f"Std evaluation: {dataset.evaluations.std().item()}")

    # Plot the evaluation distribution
    plt.hist(dataset.evaluations, bins=100)
    plt.xlabel('Evaluation')
    plt.ylabel('Count')
    plt.title('Evaluation Distribution')
    plt.show()
