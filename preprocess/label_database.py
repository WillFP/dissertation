import multiprocessing as mp
from itertools import islice
from pathlib import Path
import json
import h5py
import numpy as np
from tqdm import tqdm
import chess
from preprocess.encode import fen_to_tensor

# Configuration
NUM_WORKERS = mp.cpu_count()
CHUNK_SIZE = 10000


def compute_evaluation(position_data):
    """
    Compute the evaluation score from Stockfish data for a given position.
    - Mate scores reflect moves to mate, with closer mates stronger.
    - cp scores are clipped to [-99, 99].
    - Returns a score from White's perspective, within [-100, 100].
    """
    evals = position_data["evals"]
    if not evals:
        raise ValueError("No evaluations for position")

    max_knodes_eval = max(evals, key=lambda e: e["knodes"])
    first_pv = max_knodes_eval["pvs"][0]

    if "mate" in first_pv:
        mate = first_pv["mate"]
        if mate > 0:
            n = mate
            evaluation = 100 - 1 * (n - 1)  # e.g., mate in 1 = 99.99, mate in 5 = 99.95
        else:
            n = -mate
            evaluation = -(100 - 1 * (n - 1))

    else:
        cp = first_pv["cp"]
        evaluation = cp / 100.0
        evaluation = np.clip(evaluation, -75, 75)

    return evaluation


def position_reader(json_path, max_positions=None):
    """
    Generator that reads positions from a JSON file.
    Yields one position's data per iteration.
    """
    with open(json_path, "r") as f:
        for i, line in enumerate(f):
            if max_positions and i >= max_positions:
                break
            position_data = json.loads(line)
            yield position_data


def worker_process(position_chunk):
    """
    Process a chunk of positions:
    - Computes evaluations using Stockfish data.
    - Converts FENs to tensors and metadata.
    - Returns results in the required format.
    """
    results = {
        'boards': [],
        'metadata': [],
        'evaluations': []
    }
    for position_data in position_chunk:
        fen = position_data["fen"]
        try:
            evaluation = compute_evaluation(position_data)
            tensor, meta = fen_to_tensor(fen)
            results['boards'].append(tensor)
            results['metadata'].append(meta)
            results['evaluations'].append(evaluation)
        except Exception as e:
            print(f"Error processing position {fen}: {e}")
    return results


def merge_results(temp_files, output_file):
    """
    Merge temporary HDF5 files into a single final file.
    Maintains the same structure as the original MCTS output.
    """
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('boards', shape=(0, 8, 8, 12),
                          maxshape=(None, 8, 8, 12),
                          dtype=np.uint8, compression='gzip')
        hf.create_dataset('metadata', shape=(0, 5),
                          maxshape=(None, 5),
                          dtype=np.uint8, compression='gzip')
        hf.create_dataset('evaluations', shape=(0,),
                          maxshape=(None,),
                          dtype=np.float32, compression='gzip')
        for temp_file in tqdm(temp_files, desc="Merging results"):
            with h5py.File(temp_file, 'r') as src:
                num_samples = src['boards'].shape[0]
                hf['boards'].resize(hf['boards'].shape[0] + num_samples, axis=0)
                hf['metadata'].resize(hf['metadata'].shape[0] + num_samples, axis=0)
                hf['evaluations'].resize(hf['evaluations'].shape[0] + num_samples, axis=0)
                hf['boards'][-num_samples:] = src['boards'][:]
                hf['metadata'][-num_samples:] = src['metadata'][:]
                hf['evaluations'][-num_samples:] = src['evaluations'][:]
            Path(temp_file).unlink()


def main(json_path, hdf5_path, positions):
    """
    Main function:
    - Reads JSON position data.
    - Processes positions in parallel using Stockfish evaluations.
    - Saves results to an HDF5 file.
    """
    temp_dir = Path("temp_labels")
    temp_dir.mkdir(exist_ok=True)

    position_gen = position_reader(json_path, max_positions=positions)

    with mp.Pool(processes=NUM_WORKERS) as pool:
        def chunk_generator():
            while True:
                chunk = list(islice(position_gen, CHUNK_SIZE))
                if not chunk:
                    break
                yield chunk

        temp_files = []
        chunk_count = 0
        with tqdm(desc="Processing position chunks", unit="chunk") as pbar:
            for result in pool.imap_unordered(worker_process, chunk_generator(), chunksize=1):
                chunk_count += 1
                pbar.update(1)
                if not result['boards']:
                    continue
                temp_file = temp_dir / f"temp_{chunk_count}.h5"
                with h5py.File(temp_file, 'w') as hf:
                    hf.create_dataset('boards', data=result['boards'], dtype=np.uint8)
                    hf.create_dataset('metadata', data=result['metadata'], dtype=np.uint8)
                    hf.create_dataset('evaluations', data=result['evaluations'], dtype=np.float32)
                temp_files.append(str(temp_file))
                pbar.set_postfix({'positions': len(result['boards'])})

        merge_results(temp_files, hdf5_path)

    print(f"Completed labeling. Final dataset: {hdf5_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Label chess positions with Stockfish evaluations')
    parser.add_argument('--json', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 file path')
    parser.add_argument('--positions', type=int, required=True, help='Total positions to process')
    args = parser.parse_args()
    main(args.json, args.output, args.positions)
