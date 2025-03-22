import multiprocessing as mp
from itertools import islice
from pathlib import Path

import chess
import chess.pgn
import h5py
import numpy as np
from tqdm import tqdm

from preprocess.encode import fen_to_tensor

# Configuration
POSITIONS_PER_GAME = 10 # Use higher positions per game compared to stockfish to capture more end-game
RANDOM_SEED = 42
NUM_WORKERS = mp.cpu_count()
CHUNK_SIZE = 5  # Games per batch

def mcts_evaluation(fen, rng, num_simulations=150, max_moves=100):
    board = chess.Board(fen)

    white_wins = 0
    black_wins = 0

    for _ in range(num_simulations):
        move_count = 0

        while not board.is_game_over() and move_count < max_moves:
            legal_moves = list(board.legal_moves)
            move = legal_moves[rng.integers(0, len(legal_moves))]
            board.push(move)
            move_count += 1

        result = board.result()

        if result == "1-0":
            white_wins += 1
        elif result == "0-1":
            black_wins += 1

        for _ in range(move_count):
            board.pop()

    p_white_win = white_wins / num_simulations
    p_black_win = black_wins / num_simulations

    return (p_white_win - p_black_win) * 100


def extract_positions(game):
    """
    Extract FEN positions from a game (mainline),
    skipping captures and only including positions after the first five moves.
    """
    positions = []
    board = game.board()
    move_count = 0
    for move in game.mainline_moves():
        is_capture = board.is_capture(move)
        board.push(move)
        move_count += 1
        if move_count > 5 and not is_capture:
            positions.append(board.fen())
    return positions


def game_reader(pgn_path, max_games=None):
    """
    Generator that reads games from PGN.
    Yields a list of position FENs for each game.
    """
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as pgn:
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


def init_worker():
    """
    Initializer for each worker process:
      - Sets up a unique RNG seed per worker
    """
    global rng
    worker_id = mp.current_process()._identity[0]
    rng = np.random.default_rng(RANDOM_SEED + worker_id)


def worker_process(positions_chunk):
    """
    Worker process function that:
      - Takes a list of game-positions
      - Selects random positions
      - Evaluates them with MCTS
      - Converts each to a tensor
      - Returns aggregated results
    """
    global rng
    results = {
        'boards': [],
        'metadata': [],
        'evaluations': []
    }
    for positions in positions_chunk:
        if len(positions) < POSITIONS_PER_GAME:
            continue
        selected_fens = rng.choice(positions, POSITIONS_PER_GAME, replace=False)
        for fen in selected_fens:
            evaluation = mcts_evaluation(fen, rng)
            tensor, meta = fen_to_tensor(fen)
            results['boards'].append(tensor)
            results['metadata'].append(meta)
            results['evaluations'].append(evaluation)
    return results


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
            Path(temp_file).unlink()


def main(pgn_path, hdf5_path, positions):
    """
    Main function:
      - Reads PGN
      - Distributes tasks to workers
      - Collects results
      - Merges into final HDF5
    """
    temp_dir = Path("temp_labels")
    temp_dir.mkdir(exist_ok=True)
    max_games = positions // POSITIONS_PER_GAME if positions else None
    game_gen = game_reader(pgn_path, max_games=max_games)
    with mp.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        def chunk_generator():
            while True:
                chunk = list(islice(game_gen, CHUNK_SIZE))
                if not chunk:
                    break
                yield chunk

        temp_files = []
        chunk_iter = pool.imap_unordered(worker_process, chunk_generator(), chunksize=1)
        chunk_count = 0
        with tqdm(desc="Processing game chunks with MCTS", unit="chunk") as pbar:
            for result in chunk_iter:
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

    parser = argparse.ArgumentParser(description='Label chess positions with MCTS evaluations')
    parser.add_argument('--pgn', type=str, required=True, help='Input PGN file path')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 file path')
    parser.add_argument('--positions', type=int, required=True, help='Total positions to sample')
    args = parser.parse_args()
    main(args.pgn, args.output, args.positions)
