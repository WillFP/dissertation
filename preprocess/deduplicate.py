import argparse

import h5py
from tqdm import tqdm


def deduplicate_chess_dataset(input_path, output_path, batch_size=10000):
    """
    Deduplicate a chess dataset by removing duplicate board positions.

    Parameters:
    - input_path: str, path to the input .h5 file
    - output_path: str, path to the output .h5 file
    - batch_size: int, number of samples to process at once
    """
    # Open the input .h5 file in read mode
    with h5py.File(input_path, 'r') as f_in:
        boards = f_in['boards'][:]
        metadata = f_in['metadata'][:]
        evaluations = f_in['evaluations'][:]

        if evaluations.ndim == 2 and evaluations.shape[1] == 1:
            evaluations = evaluations.squeeze(1)

        print(f"Original shapes: boards {boards.shape}, metadata {metadata.shape}, evaluations {evaluations.shape}")

        seen_boards = {}
        unique_indices = []
        total_samples = boards.shape[0]

        print(f"Identifying unique positions...")
        for start_idx in tqdm(range(0, total_samples, batch_size)):
            end_idx = min(start_idx + batch_size, total_samples)

            batch_boards = boards[start_idx:end_idx]

            for i, board in enumerate(batch_boards):
                idx = start_idx + i
                board_hash = hash(board.tobytes())

                if board_hash not in seen_boards:
                    seen_boards[board_hash] = idx
                    unique_indices.append(idx)

        unique_indices.sort()

        dedup_boards = boards[unique_indices]
        dedup_metadata = metadata[unique_indices]
        dedup_evaluations = evaluations[unique_indices]

        print(
            f"Deduplicated shapes: boards {dedup_boards.shape}, metadata {dedup_metadata.shape}, evaluations {dedup_evaluations.shape}")
        print(
            f"Removed {total_samples - len(unique_indices)} duplicate positions ({(total_samples - len(unique_indices)) / total_samples:.2%})")

        with h5py.File(output_path, 'w') as f_out:
            f_out.create_dataset('boards', data=dedup_boards, dtype='float32', compression='gzip')
            f_out.create_dataset('metadata', data=dedup_metadata, dtype='float32', compression='gzip')
            f_out.create_dataset('evaluations', data=dedup_evaluations, dtype='float32', compression='gzip')

        print(f"Deduplicated dataset saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deduplicate chess positions in an HDF5 dataset')
    parser.add_argument('--input', type=str, required=True, help='Input HDF5 file path')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 file path')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for processing')
    args = parser.parse_args()

    deduplicate_chess_dataset(args.input, args.output, args.batch_size)
