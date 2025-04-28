import h5py
import numpy as np


def augment_board(board):
    # flip vertically (reverse rows)
    flipped = board[:, ::-1, :, :]
    # swap colors: channels 0-5 (white) to 6-11 (black) and vice versa
    augmented = flipped[:, :, :, [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]
    return augmented


def augment_metadata(metadata):
    # Swap castling rights and invert turn
    augmented = np.column_stack([
        metadata[:, 2],  # black_kingside
        metadata[:, 3],  # black_queenside
        metadata[:, 0],  # white_kingside
        metadata[:, 1],  # white_queenside
        1 - metadata[:, 4]  # inverted turn
    ])
    return augmented


def augment_evaluation(evaluation):
    return -evaluation


def create_augmented_h5(input_path, output_path):
    """
    Create a new .h5 file containing both original and augmented chess positions.

    Parameters:
    - input_path: str, path to the input .h5 file
    - output_path: str, path to the output .h5 file
    """
    # Open the input .h5 file in read mode
    with h5py.File(input_path, 'r') as f_in:
        boards = f_in['boards'][:]
        metadata = f_in['metadata'][:]
        evaluations = f_in['evaluations'][:]

        if evaluations.ndim == 2 and evaluations.shape[1] == 1:
            evaluations = evaluations.squeeze(1)

        print(f"Original shapes: boards {boards.shape}, metadata {metadata.shape}, evaluations {evaluations.shape}")

        aug_boards = augment_board(boards)
        aug_metadata = augment_metadata(metadata)
        aug_evaluations = augment_evaluation(evaluations)

        all_boards = np.concatenate([boards, aug_boards], axis=0)
        all_metadata = np.concatenate([metadata, aug_metadata], axis=0)
        all_evaluations = np.concatenate([evaluations, aug_evaluations], axis=0)

        print(
            f"Augmented shapes: boards {all_boards.shape}, metadata {all_metadata.shape}, evaluations {all_evaluations.shape}")

        with h5py.File(output_path, 'w') as f_out:
            f_out.create_dataset('boards', data=all_boards, dtype='float32', compression='gzip')
            f_out.create_dataset('metadata', data=all_metadata, dtype='float32', compression='gzip')
            f_out.create_dataset('evaluations', data=all_evaluations, dtype='float32', compression='gzip')

        print(f"Augmented dataset saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Augment chess positions in an HDF5 dataset')
    parser.add_argument('--input', type=str, required=True, help='Input HDF5 file path')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 file path')
    args = parser.parse_args()

    create_augmented_h5(args.input, args.output)
