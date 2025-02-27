import torch

from preprocess import fen_to_tensor

import chess
from typing import Optional
from modeling.autoencoder.predict import load_model as load_autoencoder
from modeling.evaluation.predict import load_model as load_evaluator, predict_position


def load_models(autoencoder_path: str, evaluation_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f"Loading autoencoder from {autoencoder_path}...")
    autoencoder = load_autoencoder(autoencoder_path, latent_dim=128, device=device)

    print(f"Loading evaluator from {evaluation_path}...")
    evaluator = load_evaluator(evaluation_path, latent_dim=128, device=device)

    print("Models loaded successfully.")

    return autoencoder, evaluator


def predict_fen(fen, autoencoder, evaluator):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    board, metadata = fen_to_tensor(fen)
    evaluation = predict_position(
        evaluator,
        autoencoder,
        torch.FloatTensor(board).permute(2, 0, 1),
        torch.FloatTensor(metadata),
        device=device
    )
    score = evaluation.item()

    return score


def get_best_move(fen: str, is_white: bool, autoencoder, evaluator) -> Optional[chess.Move]:
    """Minimax with alpha-beta pruning using only neural network evaluations."""
    board = chess.Board(fen)
    if not board.legal_moves:
        return None

    depth = 3
    transposition_table = {}

    def order_moves(board: chess.Board) -> list[chess.Move]:
        """Order moves using game rules without any position evaluation."""
        return sorted(
            board.legal_moves,
            key=lambda m: (
                board.is_capture(m),  # Captures first
                bool(m.promotion),  # Promotions next
                board.is_check()  # Checks next
            ),
            reverse=True
        )

    def alpha_beta(b: chess.Board, d: int, alpha: float, beta: float) -> float:
        fen_key = b.fen()
        if fen_key in transposition_table:
            return transposition_table[fen_key]

        if d == 0 or b.is_game_over():
            return predict_fen(fen_key, autoencoder, evaluator)

        is_maximizing = b.turn == is_white
        best = -float('inf') if is_maximizing else float('inf')

        for move in order_moves(b):
            b.push(move)
            eval = alpha_beta(b, d - 1, alpha, beta)
            b.pop()

            if is_maximizing:
                best = max(best, eval)
                alpha = max(alpha, best)
            else:
                best = min(best, eval)
                beta = min(beta, best)

            if beta <= alpha:
                break

        transposition_table[fen_key] = best
        return best

    best_move = None
    best_eval = -float('inf') if is_white else float('inf')
    alpha, beta = -float('inf'), float('inf')

    for move in order_moves(board):
        board.push(move)
        eval = alpha_beta(board, depth - 1, alpha, beta)
        board.pop()

        print(f"Move {move.uci()} score: {eval:.3f}")

        if (is_white and eval > best_eval) or (not is_white and eval < best_eval):
            best_eval = eval
            best_move = move
            print(f"New best: {move.uci()} ({eval:.3f})")

            if is_white:
                alpha = max(alpha, eval)
            else:
                beta = min(beta, eval)

            if alpha >= beta:
                break

    return best_move
