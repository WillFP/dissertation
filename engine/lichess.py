import berserk
import chess
import time
from typing import Optional

from engine.bot import predict_fen


class LichessBot:
    def __init__(self, api_token: str):
        """Initialize the Lichess bot with the provided API token."""
        self.session = berserk.TokenSession(api_token)
        self.client = berserk.Client(self.session)
        self.current_game: Optional[str] = None
        self.color: Optional[chess.Color] = None

    def get_best_move(self, fen: str, is_white: bool) -> Optional[chess.Move]:
        """Calculate the best move for the current position using the prediction function."""
        board = chess.Board(fen)

        if not board.legal_moves:
            return None

        best_move = None
        best_score = -float('inf') if is_white else float('inf')

        for move in board.legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            score = predict_fen(board_copy.fen())
            print(f"Move {move.uci()} has score {score} for {'white' if is_white else 'black'}")

            if (is_white and score > best_score) or (not is_white and score < best_score):
                best_score = score
                best_move = move
                print(f"New best move: {move.uci()}, score: {score}")

        return best_move

    def handle_game_stream(self):
        """Main loop to handle incoming game events and challenges."""
        while True:
            try:
                for event in self.client.bots.stream_incoming_events():
                    if event['type'] == 'challenge':
                        self.client.bots.accept_challenge(event['challenge']['id'])
                    elif event['type'] == 'gameStart':
                        self.current_game = event['game']['id']
                        self.play_game(event['game']['id'])
            except Exception as e:
                print(f"Error in game stream: {e}")
                time.sleep(5)

    def play_game(self, game_id: str):
        """Manage gameplay for a single game."""
        try:
            for event in self.client.bots.stream_game_state(game_id):
                if event['type'] == 'gameFull':
                    self._handle_game_full_event(event, game_id)
                elif event['type'] == 'gameState':
                    self._handle_game_state_event(event, game_id)
        except Exception as e:
            print(f"Error in game {game_id}: {e}")

    def _handle_game_full_event(self, event: dict, game_id: str):
        """Handle initial game state when a game starts."""
        bot_id = self.client.account.get()['id']
        self.color = chess.WHITE if event['white']['id'] == bot_id else chess.BLACK

        if not event['state']['moves'] and self.color == chess.WHITE:
            self.make_move(game_id, chess.STARTING_FEN, is_white=True)

    def _handle_game_state_event(self, event: dict, game_id: str):
        """Handle ongoing game state updates."""
        board = self._reconstruct_board(event['moves'])

        if board.turn == self.color and event['status'] == 'started':
            self.make_move(game_id, board.fen(), is_white=(self.color == chess.WHITE))

    def _reconstruct_board(self, moves_str: str) -> chess.Board:
        """Reconstruct a chess board from a moves string."""
        board = chess.Board()
        if moves_str:
            for move in moves_str.split():
                board.push_uci(move)
        return board

    def make_move(self, game_id: str, fen: str, is_white: bool):
        """Calculate and execute the best move for the current position."""
        try:
            move = self.get_best_move(fen, is_white)
            if not move:
                print("No legal moves available.")
                return

            self.client.bots.make_move(game_id, move.uci())
            print(f"Made move {move.uci()} in game {game_id}")
        except Exception as e:
            print(f"Error making move: {e}")


def main():
    """Entry point to start the bot."""
    api_token = "lip_jDdpzsUIuxZhy1gIKgig"  # Replace with your actual token
    bot = LichessBot(api_token)

    while True:
        try:
            bot.handle_game_stream()
        except Exception as e:
            print(f"Critical error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
