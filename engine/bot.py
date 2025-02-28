import torch
from preprocess import fen_to_tensor
import chess
from typing import Optional
from modeling.autoencoder.predict import load_model as load_autoencoder
from modeling.evaluation.predict import load_model as load_evaluator, predict_position


def load_models(autoencoder_path: str, evaluation_path: str):
    """
    Load the autoencoder and evaluator models onto the appropriate device.

    Args:
        autoencoder_path (str): Path to the trained autoencoder model file.
        evaluation_path (str): Path to the trained evaluator model file.

    Returns:
        tuple: Loaded autoencoder and evaluator models.
    """
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading autoencoder from {autoencoder_path}...")
    autoencoder = load_autoencoder(autoencoder_path, latent_dim=128, device=device)
    print(f"Loading evaluator from {evaluation_path}...")
    evaluator = load_evaluator(evaluation_path, latent_dim=128, device=device)
    print("Models loaded successfully.")
    return autoencoder, evaluator


def predict_fen(fen, autoencoder, evaluator):
    """
    Evaluate a chess position using the CNN, returning a score from White's perspective.

    Args:
        fen (str): The position in FEN notation.
        autoencoder: The loaded autoencoder model.
        evaluator: The loaded evaluator model.

    Returns:
        float: Evaluation score from White's perspective.
    """
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    board, metadata = fen_to_tensor(fen)
    evaluation = predict_position(
        evaluator,
        autoencoder,
        torch.FloatTensor(board).permute(2, 0, 1),
        torch.FloatTensor(metadata),
        device=device
    )
    return evaluation.item()


def get_best_move(fen: str, autoencoder, evaluator) -> Optional[chess.Move]:
    """
    Find the best move for the side to move in the given FEN using minimax with alpha-beta pruning,
    relying solely on CNN evaluations from White's perspective.

    Args:
        fen (str): The position in FEN notation, where the side to move is the engine's side.
        autoencoder: The loaded autoencoder model.
        evaluator: The loaded evaluator model.

    Returns:
        Optional[chess.Move]: The best move, or None if no legal moves exist.
    """
    board = chess.Board(fen)
    if not board.legal_moves:
        return None

    depth = 3  # Search depth
    transposition_table = {}  # For memoization

    def alpha_beta(b: chess.Board, d: int, alpha: float, beta: float) -> float:
        """
        Perform minimax search with alpha-beta pruning, where evaluation is from White's perspective.

        Args:
            b (chess.Board): Current board position.
            d (int): Remaining depth.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.

        Returns:
            float: Evaluation score from White's perspective.
        """
        fen_key = b.fen()
        if fen_key in transposition_table:
            return transposition_table[fen_key]

        if d == 0 or b.is_game_over():
            eval = predict_fen(fen_key, autoencoder, evaluator)
            transposition_table[fen_key] = eval
            return eval

        if b.turn == chess.WHITE:
            best = -float('inf')  # Maximize for White
            for move in b.legal_moves:
                b.push(move)
                eval = alpha_beta(b, d - 1, alpha, beta)
                b.pop()
                best = max(best, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            transposition_table[fen_key] = best
            return best
        else:  # b.turn == chess.BLACK
            best = float('inf')  # Minimize for Black
            for move in b.legal_moves:
                b.push(move)
                eval = alpha_beta(b, d - 1, alpha, beta)
                b.pop()
                best = min(best, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            transposition_table[fen_key] = best
            return best

    alpha = -float('inf')
    beta = float('inf')
    best_move = None

    if board.turn == chess.WHITE:
        best_eval = -float('inf')  # White maximizes
        for move in board.legal_moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta)
            board.pop()
            print(f"Move {move.uci()} score: {eval:.3f}")
            if eval > best_eval:
                best_eval = eval
                best_move = move
                print(f"New best: {move.uci()} ({eval:.3f})")
            alpha = max(alpha, eval)
    else:  # board.turn == chess.BLACK
        best_eval = float('inf')  # Black minimizes
        for move in board.legal_moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta)
            board.pop()
            print(f"Move {move.uci()} score: {eval:.3f}")
            if eval < best_eval:
                best_eval = eval
                best_move = move
                print(f"New best: {move.uci()} ({eval:.3f})")
            beta = min(beta, eval)

    return best_move
