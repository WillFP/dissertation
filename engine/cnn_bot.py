import torch

from engine import ChessBot
from modeling.predict import load_model as load_evaluator
from preprocess import fen_to_tensor


class CNNChessBot(ChessBot):
    def __init__(self, path: str):
        super().__init__()
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
