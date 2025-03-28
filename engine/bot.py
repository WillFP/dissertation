from typing import Optional

import chess


class ChessBot:
    def __init__(self, log=False):
        """
        Initialize the ChessBot
        """
        self.log = log

    def predict_fen(self, fen: str) -> float:
        raise NotImplementedError("Subclasses must implement this method")

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
                if self.log:
                    print(f"Move {move.uci()} score: {evaluation:.3f}")
                if evaluation > best_eval:
                    best_eval = evaluation
                    best_move = move
                    if self.log:
                        print(f"New best: {move.uci()} ({evaluation:.3f})")
                alpha = max(alpha, evaluation)
        else:  # board.turn == chess.BLACK
            best_eval = float('inf')  # Black minimizes
            for move in board.legal_moves:
                board.push(move)
                evaluation = alpha_beta(board, depth - 1, alpha, beta)
                board.pop()
                if self.log:
                    print(f"Move {move.uci()} score: {evaluation:.3f}")
                if evaluation < best_eval:
                    best_eval = evaluation
                    best_move = move
                    if self.log:
                        print(f"New best: {move.uci()} ({evaluation:.3f})")
                beta = min(beta, evaluation)

        return best_move
