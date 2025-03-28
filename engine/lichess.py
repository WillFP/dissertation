import argparse
import time
from typing import Optional

import berserk
import chess

from engine.cnn_bot import CNNChessBot


class LichessBot:
    def __init__(self, api_token: str, model_path: str):
        """Initialize the Lichess bot with the provided API token."""
        self.session = berserk.TokenSession(api_token)
        self.client = berserk.Client(self.session)
        self.current_game: Optional[str] = None
        self.color: Optional[chess.Color] = None
        self.engine = CNNChessBot(model_path)

    def handle_game_stream(self):
        """Main loop to handle incoming game events and challenges."""
        while True:
            try:
                for event in self.client.bots.stream_incoming_events():
                    print(f"Received event: {event}")
                    if event['type'] == 'challenge':
                        self.client.bots.accept_challenge(event['challenge']['id'])
                    elif event['type'] == 'gameStart':
                        print(f"Starting game {event['game']['id']}")
                        self.current_game = event['game']['id']
                        self.play_game(event['game']['id'])
                    elif event['type'] == 'gameFinish':
                        print(f"Game {event['game']['id']} finished")
                        self.current_game = None
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
            self.make_move(game_id, chess.STARTING_FEN)

    def _handle_game_state_event(self, event: dict, game_id: str):
        """Handle ongoing game state updates."""
        board = self._reconstruct_board(event['moves'])

        if board.turn == self.color and event['status'] == 'started':
            self.make_move(game_id, board.fen())

    def _reconstruct_board(self, moves_str: str) -> chess.Board:
        """Reconstruct a chess board from a moves string."""
        board = chess.Board()
        if moves_str:
            for move in moves_str.split():
                board.push_uci(move)
        return board

    def make_move(self, game_id: str, fen: str):
        """Calculate and execute the best move for the current position."""
        try:
            move = self.engine.get_best_move(fen)
            if not move:
                print("No legal moves available.")
                return

            self.client.bots.make_move(game_id, move.uci())
            print(f"Made move {move.uci()} in game {game_id}")
        except Exception as e:
            print(f"Error making move: {e}")


def main():
    """Entry point to start the bot."""

    parser = argparse.ArgumentParser(description='Run a bot on Lichess')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the model')

    args = parser.parse_args()

    api_token = "lip_uOVXFaLVX99unHcNavZl"
    bot = LichessBot(api_token, args.model)

    while True:
        try:
            bot.handle_game_stream()
        except Exception as e:
            print(f"Critical error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
