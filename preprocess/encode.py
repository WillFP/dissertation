import chess
import numpy as np


def fen_to_tensor(fen):
    """Convert FEN string to board tensor and metadata array"""
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 12), dtype=np.uint8)

    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            tensor[row, col, piece_to_channel[piece.symbol()]] = 1

    # Metadata: castling rights + active color
    castling = board.castling_rights
    metadata = np.array([
        bool(castling & chess.BB_H1),  # White kingside
        bool(castling & chess.BB_A1),  # White queenside
        bool(castling & chess.BB_H8),  # Black kingside
        bool(castling & chess.BB_A8),  # Black queenside
        board.turn  # 1 for white, 0 for black
    ], dtype=np.uint8)

    return tensor, metadata
