import chess
import chess.engine
import chess.pgn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Configuration
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # Adjust this path if needed
POSITIONS_PER_GAME = 10
RANDOM_SEED = 42


def stockfish_evaluation(board, engine):
    """Get Stockfish evaluation in pawns (white perspective)."""
    info = engine.analyse(board, chess.engine.Limit(depth=15))
    score = info['score'].white().score(mate_score=10000)
    return score / 100.0  # Convert from centipawns to pawns


def mcts_evaluation(fen, rng, num_simulations=300, max_moves=50):
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
    """Extract FEN positions from a game, skipping captures and moves before 6."""
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


def main(pgn_path, positions):
    """Compare MCTS and Stockfish evaluations with progress tracking."""
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # Initialize Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    # Lists to store evaluations
    stockfish_evals = []
    mcts_evals = []

    # Process PGN file with tqdm progress bar
    game_count = 0
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as pgn:
        with tqdm(total=positions, desc="Collecting positions", unit="position") as pbar:
            while len(stockfish_evals) < positions:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    print("Reached end of PGN file.")
                    break
                positions_list = extract_positions(game)
                if len(positions_list) >= POSITIONS_PER_GAME:
                    selected_fens = rng.choice(positions_list, POSITIONS_PER_GAME, replace=False)
                    for fen in selected_fens:
                        board = chess.Board(fen)
                        sf_eval = stockfish_evaluation(board, engine)
                        mcts_eval = mcts_evaluation(fen, rng)

                        if 90 <= sf_eval:
                            print(f"Near-checkmate: stockfish predicts {sf_eval:.2f}, mcts predicts {mcts_eval:.2f}")
                        stockfish_evals.append(sf_eval)
                        mcts_evals.append(mcts_eval)
                        pbar.update(1)
                        if len(stockfish_evals) >= positions:
                            break
                game_count += 1
                pbar.set_postfix({'games': game_count})  # Show games processed

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(stockfish_evals, mcts_evals, alpha=0.5)
    plt.xlabel('Stockfish Evaluation')
    plt.ylabel('MCTS Evaluation')
    plt.title('Comparison of Stockfish and MCTS Evaluations')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.show()

    # Compute and print correlation
    if stockfish_evals and mcts_evals:
        correlation = np.corrcoef(stockfish_evals, mcts_evals)[0, 1]
        print(f"Pearson correlation coefficient: {correlation:.3f}")
    else:
        print("No positions were collected.")

    # Clean up
    engine.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare MCTS and Stockfish evaluations with progress')
    parser.add_argument('--pgn', type=str, required=True, help='Input PGN file path')
    parser.add_argument('--positions', type=int, required=True, help='Total positions to sample')
    args = parser.parse_args()
    main(args.pgn, args.positions)
