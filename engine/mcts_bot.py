import chess
import numpy as np

from engine import ChessBot


class MCTSChessBot(ChessBot):
    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng(42)

    def simulate_game(self, board, max_moves=100):
        """
        Simulate a game from the current board position until the end or max_moves.
        Returns the game result: '1-0' (white wins), '0-1' (black wins), or '1/2-1/2' (draw).
        """
        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return "1/2-1/2"
            move = self.rng.choice(legal_moves)
            board.push(move)
            move_count += 1
        if board.is_game_over():
            return board.result()
        else:
            return "1/2-1/2"  # Treat as draw if max_moves reached

    def mcts_evaluation(self, fen, num_simulations=200, max_moves=200):
        """
        Evaluate a position using MCTS with random simulations.
        Returns the evaluation score as p_white_win - p_black_win, ranging from -100 to 100.
        """
        board = chess.Board(fen)
        white_wins = 0
        black_wins = 0
        for _ in range(num_simulations):
            temp_board = board.copy()
            result = self.simulate_game(temp_board, max_moves)
            if result == "1-0":
                white_wins += 1
            elif result == "0-1":
                black_wins += 1
        p_white_win = white_wins / num_simulations
        p_black_win = black_wins / num_simulations
        evaluation = p_white_win - p_black_win
        return evaluation * 100

    def predict_fen(self, fen: str) -> float:
        """
        Evaluate a chess position given in FEN notation.

        Args:
            fen: FEN string representing the chess position

        Returns:
            Evaluation score as a float
        """
        return self.mcts_evaluation(fen, num_simulations=50, max_moves=50)
