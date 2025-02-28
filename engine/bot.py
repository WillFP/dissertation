import torch
from preprocess import fen_to_tensor
import chess
from typing import Optional
from modeling.autoencoder.predict import load_model as load_autoencoder
from modeling.evaluation.predict import load_model as load_evaluator, predict_position


def load_models(autoencoder_path: str, evaluation_path: str):
    """Load the autoencoder and evaluator models onto the appropriate device."""
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading autoencoder from {autoencoder_path}...")
    autoencoder = load_autoencoder(autoencoder_path, latent_dim=128, device=device)
    print(f"Loading evaluator from {evaluation_path}...")
    evaluator = load_evaluator(evaluation_path, latent_dim=128, device=device)
    print("Models loaded successfully.")
    return autoencoder, evaluator


def predict_fen(fen, autoencoder, evaluator):
    """Evaluate a chess position using the CNN, returning a score from the current player's perspective."""
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
    relying solely on CNN evaluations.

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

    depth = 3
    transposition_table = {}

    def alpha_beta(b: chess.Board, d: int, alpha: float, beta: float) -> float:
        """
        Perform minimax search with alpha-beta pruning, evaluating from the current player's perspective.

        Args:
            b (chess.Board): The current board position.
            d (int): Remaining depth.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.

        Returns:
            float: The evaluation score from the current player's perspective.
        """
        fen_key = b.fen()
        if fen_key in transposition_table:
            return transposition_table[fen_key]

        if d == 0 or b.is_game_over():
            return predict_fen(fen_key, autoencoder, evaluator)

        best = -float('inf')
        for move in b.legal_moves:  # No move ordering
            b.push(move)
            eval = -alpha_beta(b, d - 1, -beta, -alpha)  # Negate to switch perspective
            b.pop()
            best = max(best, eval)
            alpha = max(alpha, eval)
            if alpha >= beta:
                break
        transposition_table[fen_key] = best
        return best

    alpha = -float('inf')
    beta = float('inf')
    best_move = None
    best_eval = -float('inf')

    for move in board.legal_moves:  # No move ordering
        board.push(move)
        opponent_eval = alpha_beta(board, depth - 1, -beta, -alpha)
        board.pop()
        our_eval = -opponent_eval  # Negate to get the engine's perspective
        print(f"Move {move.uci()} score: {our_eval:.3f}")
        if our_eval > best_eval:
            best_eval = our_eval
            best_move = move
            print(f"New best: {move.uci()} ({our_eval:.3f})")
        alpha = max(alpha, our_eval)
        if alpha >= beta:
            break

    return best_move
