from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def merge_results(temp_files, output_file):
    """
    Merge the temporary HDF5 files into a final single file.
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


def main(temp_dir_path, hdf5_path):
    """
    Main function:
      - Reads PGN
      - Distributes tasks to workers
      - Collects results
      - Merges into final HDF5
    """
    temp_dir = Path(temp_dir_path)
    temp_dir.mkdir(exist_ok=True)
    temp_files = list(temp_dir.glob("*.h5"))
    merge_results(temp_files, hdf5_path)
    print(f"Merged temp files into {hdf5_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Label chess positions with MCTS evaluations')
    parser.add_argument('--path', type=str, required=True, help='Path for .h5 files')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 file path')
    args = parser.parse_args()
    main(args.path, args.output)
