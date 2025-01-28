import torch

from preprocess import fen_to_tensor

import chess
from typing import Optional
from modeling.autoencoder.predict import load_model as load_autoencoder
from modeling.evaluation.predict import load_model as load_evaluator, predict_position

evaluation_model_path = "models/best_eval_model.pt"
autoencoder_model_path = "models/best_autoencoder.pt"

def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    autoencoder = load_autoencoder(autoencoder_model_path, latent_dim=128, device=device)
    evaluator = load_evaluator(evaluation_model_path, latent_dim=128, device=device)

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
    """Calculate the best move using optimized minimax with alpha-beta pruning."""
    board = chess.Board(fen)

    if not board.legal_moves:
        return None

    depth = 3  # Adjust depth based on performance needs

    def order_moves(board: chess.Board) -> list[chess.Move]:
        """Basic move ordering: captures first, then non-captures."""
        return sorted(
            board.legal_moves,
            key=lambda m: board.is_capture(m),
            reverse=True
        )

    def alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        if depth == 0 or board.is_game_over():
            return predict_fen(board.fen(), autoencoder, evaluator)

        is_maximizing = (board.turn == is_white)
        best_value = -float('inf') if is_maximizing else float('inf')

        for move in order_moves(board):
            board.push(move)
            current_eval = alpha_beta(board, depth - 1, alpha, beta)
            board.pop()

            if is_maximizing:
                best_value = max(best_value, current_eval)
                alpha = max(alpha, best_value)
            else:
                best_value = min(best_value, current_eval)
                beta = min(beta, best_value)

            if beta <= alpha:
                break

        return best_value

    best_move = None
    best_value = -float('inf') if is_white else float('inf')
    alpha = -float('inf')
    beta = float('inf')

    for move in order_moves(board):
        board.push(move)
        current_eval = alpha_beta(board, depth - 1, alpha, beta)
        board.pop()

        print(f"Move {move.uci()} has score {current_eval} for {'white' if is_white else 'black'}")

        if is_white:
            if current_eval > best_value:
                best_value = current_eval
                best_move = move
                print(f"New best: {move.uci()} ({current_eval})")
            alpha = max(alpha, best_value)
        else:
            if current_eval < best_value:
                best_value = current_eval
                best_move = move
                print(f"New best: {move.uci()} ({current_eval})")
            beta = min(beta, best_value)

        if alpha >= beta:
            break

    return best_move
