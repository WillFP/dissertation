from typing import Optional

import chess
import torch

from modeling.predict import load_model as load_evaluator
from preprocess import fen_to_tensor


class ChessBot:
    def __init__(self, path: str):
        """
        Initialize the ChessBot with a model from the given path.

        Args:
            path: Path to the saved model
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.allocated_board = None
        self.allocated_metadata = None

        try:
            evaluator = load_evaluator(path, device=self.device)
            evaluator.eval()
            self.evaluator = evaluator
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def allocate_tensors(self, board, metadata):
        """
        Reuse tensors to avoid repeated memory allocations.

        Args:
            board: Board tensor
            metadata: Metadata tensor

        Returns:
            Tuple of allocated tensors
        """
        if self.allocated_board is None:
            self.allocated_board = board
            self.allocated_metadata = metadata
        else:
            self.allocated_board.copy_(board)
            self.allocated_metadata.copy_(metadata)

        return self.allocated_board, self.allocated_metadata

    def predict_fen(self, fen: str) -> float:
        """
        Evaluate a chess position given in FEN notation.

        Args:
            fen: FEN string representing the chess position

        Returns:
            Evaluation score as a float
        """
        try:
            board, metadata = fen_to_tensor(fen)
            board_tensor, metadata_tensor = self.allocate_tensors(
                torch.FloatTensor(board).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True),
                torch.FloatTensor(metadata).unsqueeze(0).to(self.device, non_blocking=True)
            )

            with torch.inference_mode():
                prediction = self.evaluator(board_tensor, metadata_tensor)

            return prediction.item()
        except Exception as e:
            raise ValueError(f"Error evaluating position: {e}")

    def get_best_move(self, fen: str) -> Optional[chess.Move]:
        """
        Find the best move for the side to move in the given FEN using minimax with alpha-beta pruning,
        relying solely on CNN evaluations from White's perspective.

        Args:
            fen (str): The position in FEN notation, where the side to move is the engine's side.

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
                evaluation = self.predict_fen(fen_key)
                transposition_table[fen_key] = evaluation
                return evaluation

            if b.turn == chess.WHITE:
                best = -float('inf')  # Maximize for White
                for move in b.legal_moves:
                    b.push(move)
                    evaluation = alpha_beta(b, d - 1, alpha, beta)
                    b.pop()
                    best = max(best, evaluation)
                    alpha = max(alpha, evaluation)
                    if beta <= alpha:
                        break
                transposition_table[fen_key] = best
                return best
            else:  # b.turn == chess.BLACK
                best = float('inf')  # Minimize for Black
                for move in b.legal_moves:
                    b.push(move)
                    evaluation = alpha_beta(b, d - 1, alpha, beta)
                    b.pop()
                    best = min(best, evaluation)
                    beta = min(beta, evaluation)
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
                evaluation = alpha_beta(board, depth - 1, alpha, beta)
                board.pop()
                print(f"Move {move.uci()} score: {evaluation:.3f}")
                if evaluation > best_eval:
                    best_eval = evaluation
                    best_move = move
                    print(f"New best: {move.uci()} ({evaluation:.3f})")
                alpha = max(alpha, evaluation)
        else:  # board.turn == chess.BLACK
            best_eval = float('inf')  # Black minimizes
            for move in board.legal_moves:
                board.push(move)
                evaluation = alpha_beta(board, depth - 1, alpha, beta)
                board.pop()
                print(f"Move {move.uci()} score: {evaluation:.3f}")
                if evaluation < best_eval:
                    best_eval = evaluation
                    best_move = move
                    print(f"New best: {move.uci()} ({evaluation:.3f})")
                beta = min(beta, evaluation)

        return best_move
