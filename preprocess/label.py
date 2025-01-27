import multiprocessing as mp
import sys
from itertools import islice
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import h5py
import numpy as np
from tqdm import tqdm

from encode import fen_to_tensor

# Configuration
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
EVAL_DEPTH = 18
POSITIONS_PER_GAME = 5
RANDOM_SEED = 42
MAX_GAMES = 2000  # Set to number for testing
NUM_WORKERS = mp.cpu_count() // 2  # Use half available cores
CHUNK_SIZE = 10  # Games per worker batch


def get_stockfish_evaluation(board, engine, depth=EVAL_DEPTH):
    """
    Get Stockfish evaluation in centipawns (white perspective)
    Returns: float evaluation score where positive = white advantage
    """
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info['score'].white().score(mate_score=10000)
    return score / 100.0  # Convert to pawn units


def extract_positions(game):
    """Extract FEN positions from a game"""
    positions = []
    board = game.board()

    for move in game.mainline_moves():
        board.push(move)
        positions.append(board.fen())

    return positions


def game_reader(pgn_path, max_games=None):
    """Generator that reads games and returns position lists"""
    with open(pgn_path) as pgn:
        game_count = 0
        while True:
            if max_games and game_count >= max_games:
                break

            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            positions = extract_positions(game)
            if len(positions) >= POSITIONS_PER_GAME:
                yield positions

            game_count += 1


def worker_process(args):
    """Worker process that evaluates positions and returns results"""
    (positions_chunk, worker_id) = args
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    rng = np.random.default_rng(RANDOM_SEED + worker_id)

    results = {
        'boards': [],
        'metadata': [],
        'evaluations': []
    }

    for positions in positions_chunk:
        try:
            if len(positions) < POSITIONS_PER_GAME:
                continue

            # Select random positions from game
            selected = rng.choice(positions, POSITIONS_PER_GAME, replace=False)

            for fen in selected:
                board = chess.Board(fen)
                if board.is_game_over():
                    continue

                # Get evaluation
                evaluation = get_stockfish_evaluation(board, engine)

                # Convert to tensor
                tensor, meta = fen_to_tensor(fen)

                results['boards'].append(tensor)
                results['metadata'].append(meta)
                results['evaluations'].append(evaluation)

        except Exception as e:
            print(f"Worker {worker_id} error: {str(e)}", file=sys.stderr)
            continue

    engine.quit()
    return results


def merge_results(temp_files, output_file):
    """Merge temporary HDF5 files into final dataset"""
    with h5py.File(output_file, 'w') as hf:
        # Initialize datasets
        hf.create_dataset('boards', shape=(0, 8, 8, 12), maxshape=(None, 8, 8, 12),
                          dtype=np.uint8, compression='gzip')
        hf.create_dataset('metadata', shape=(0, 5), maxshape=(None, 5),
                          dtype=np.uint8, compression='gzip')
        hf.create_dataset('evaluations', shape=(0,), maxshape=(None,),
                          dtype=np.float32, compression='gzip')

        # Merge all temp files
        for temp_file in tqdm(temp_files, desc="Merging results"):
            with h5py.File(temp_file, 'r') as src:
                num_samples = src['boards'].shape[0]

                # Resize datasets
                hf['boards'].resize(hf['boards'].shape[0] + num_samples, axis=0)
                hf['metadata'].resize(hf['metadata'].shape[0] + num_samples, axis=0)
                hf['evaluations'].resize(hf['evaluations'].shape[0] + num_samples, axis=0)

                # Append data
                hf['boards'][-num_samples:] = src['boards'][:]
                hf['metadata'][-num_samples:] = src['metadata'][:]
                hf['evaluations'][-num_samples:] = src['evaluations'][:]

            # Cleanup temp file
            Path(temp_file).unlink()


def main(pgn_path, hdf5_path):
    """Main parallel processing workflow"""
    temp_dir = Path("temp_labels")
    temp_dir.mkdir(exist_ok=True)

    # Create pool of workers
    with mp.Pool(NUM_WORKERS) as pool:
        # Create position list generator
        games = game_reader(pgn_path, MAX_GAMES)

        # Process in parallel
        tasks = ((chunk, i) for i, chunk in
                 enumerate(iter(lambda: list(islice(games, CHUNK_SIZE)), [])))

        results = pool.imap_unordered(worker_process, tasks, chunksize=1)

        # Process results with progress bar
        temp_files = []
        with tqdm(desc="Processing games", unit="chunk") as pbar:
            for i, result in enumerate(results):
                if not result['boards']:
                    continue

                # Save temporary results
                temp_file = temp_dir / f"temp_{i}.h5"
                with h5py.File(temp_file, 'w') as hf:
                    hf.create_dataset('boards', data=result['boards'], dtype=np.uint8)
                    hf.create_dataset('metadata', data=result['metadata'], dtype=np.uint8)
                    hf.create_dataset('evaluations', data=result['evaluations'], dtype=np.float32)

                temp_files.append(str(temp_file))
                pbar.update(1)
                pbar.set_postfix({'positions': sum(len(r['evaluations']) for r in [result])})

    # Merge all temporary files
    merge_results(temp_files, hdf5_path)
    print(f"Completed labeling. Final dataset: {hdf5_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Label chess positions with Stockfish evaluations')
    parser.add_argument('--pgn', type=str, required=True, help='Input PGN file path')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 file path')
    args = parser.parse_args()

    main(args.pgn, args.output)
