import argparse
import time
from typing import Optional

import berserk
import chess

from engine.cnn_bot import CNNChessBot


class LichessBot:
    def __init__(self, api_token: str, model_path: str):
        if not model_path:
            raise ValueError("A valid model path must be provided.")
        self.session = berserk.TokenSession(api_token)
        self.client = berserk.Client(self.session)
        self.current_game_id: Optional[str] = None
        self.color: Optional[chess.Color] = None
        self.engine = CNNChessBot(model_path)
        self.bot_id: Optional[str] = None

    def fetch_bot_id(self):
        if self.bot_id:
            return True
        try:
            print("Fetching account information...")
            account_info = self.client.account.get()
            if 'id' in account_info:
                self.bot_id = account_info['id']
                print(f"Successfully fetched bot ID: {self.bot_id}")
                return True
            else:
                print(f"Error: Could not retrieve bot ID from account info: {account_info}")
                return False
        except berserk.exceptions.ResponseError as e:
            print(f"API Error fetching account info: {e.status_code} {e.error}")
            if e.status_code == 401:
                print("Authentication failed. Please check your API token.")
            return False
        except Exception as e:
            print(f"Unexpected error fetching account info: {e}")
            return False

    def handle_game_stream(self):
        if not self.fetch_bot_id():
            raise RuntimeError("Could not fetch bot ID. Check API token and connection.")

        print("Starting to stream incoming events...")
        while True:
            try:
                for event in self.client.bots.stream_incoming_events():
                    if event['type'] == 'challenge':
                        challenge = event['challenge']
                        challenge_id = challenge.get('id')
                        if not challenge_id:
                            print("Warning: Received challenge without ID.")
                            continue

                        variant = challenge.get('variant', {}).get('key')
                        speed = challenge.get('speed')
                        rated = challenge.get('rated', False)

                        if variant == 'standard' and speed == 'correspondence' and rated:
                            try:
                                print(f"Accepting challenge {challenge_id}")
                                self.client.bots.accept_challenge(challenge_id)
                            except Exception as e:
                                print(f"Error accepting challenge {challenge_id}: {e}")
                        else:
                            try:
                                print(
                                    f"Declining challenge {challenge_id} (Criteria not met: v={variant}, s={speed}, r={rated})")
                                self.client.bots.decline_challenge(challenge_id)
                            except Exception as e:
                                print(f"Error declining challenge {challenge_id}: {e}")

                    elif event['type'] == 'gameStart':
                        game_info = event.get('game', {})
                        game_id = game_info.get('id')
                        if game_id:
                            # If already playing a game, ignore new game starts (or resign existing?)
                            if self.current_game_id is not None and self.current_game_id != game_id:
                                print(
                                    f"Warning: Received gameStart for {game_id} while already playing {self.current_game_id}. Ignoring new game.")
                                continue

                            print(f"Starting game {game_id}")

                            self.current_game_id = game_id
                            self.color = None
                            self.play_game(game_id)
                            print(f"Finished processing game {game_id} in handle_game_stream.")

                        else:
                            print("Error: gameStart event received without game ID.")

                    elif event['type'] == 'gameFinish':
                        game_info = event.get('game', {})
                        game_id = game_info.get('id')
                        if game_id:
                            print(f"Game {game_id} finished event received.")
                            if self.current_game_id == game_id:
                                print(f"Clearing state for finished game {game_id}.")
                                self.current_game_id = None
                                self.color = None
                        else:
                            print("Error: gameFinish event received without game ID.")

            except berserk.exceptions.ResponseError as e:
                print(f"API Response Error in event stream: {e.status_code} {e.error}")
                if e.status_code == 401:
                    print("Authentication failed. Check API token. Stopping event stream.")
                    raise
                print("Waiting 10 seconds before retrying event stream...")
                time.sleep(10)
            except Exception as e:
                print(f"Unexpected error in event stream: {e}")
                print("Waiting 5 seconds before retrying event stream...")
                time.sleep(5)

    def play_game(self, game_id: str):
        print(f"Streaming state for game {game_id}...")
        try:
            if not self.bot_id:
                print(f"Cannot play game {game_id}: Bot ID not available.")
                return

            stream = self.client.bots.stream_game_state(game_id)
            for event in stream:
                if self.current_game_id != game_id:
                    print(f"Game context switched away from {game_id} (now {self.current_game_id}). Stopping stream.")
                    break

                event_type = event.get('type')

                if event_type == 'gameFull':
                    self._handle_game_full_event(event, game_id)

                elif event_type == 'gameState':
                    if self.color is None:
                        print(f"Warning: Received gameState for {game_id} before color was determined. Ignoring state.")
                        continue

                    self._handle_game_state_event(event, game_id)

                    status = event.get('status')
                    if status not in ['created', 'started']:
                        print(f"Game {game_id} status changed to '{status}' in gameState. Ending game processing.")
                        break

        except berserk.exceptions.ResponseError as e:
            print(f"API Response Error in game stream {game_id}: {e.status_code} {e.error}")
            if e.status_code == 404:
                print(f"Game {game_id} not found.")
        except Exception as e:
            print(f"Unexpected error processing game stream for {game_id}: {e}")
        finally:
            if self.current_game_id == game_id:
                print(f"Cleaning up state for game {game_id} after stream ended/errored.")
                self.current_game_id = None
                self.color = None

    def _handle_game_full_event(self, event: dict, game_id: str):
        print(f"Handling gameFull for {game_id}.")
        try:
            if not self.bot_id:
                print(f"Error in _handle_game_full_event for {game_id}: Bot ID not available.")
                return

            white_player = event.get('white', {})
            black_player = event.get('black', {})
            white_id = white_player.get('id')
            black_id = black_player.get('id')

            if white_id == self.bot_id:
                self.color = chess.WHITE
                print(f"Bot is playing as WHITE in game {game_id}")
            elif black_id == self.bot_id:
                self.color = chess.BLACK
                print(f"Bot is playing as BLACK in game {game_id}")
            else:
                print(
                    f"CRITICAL ERROR: Bot ID {self.bot_id} not found in game {game_id}. White: {white_id}, Black: {black_id}")
                self.color = None
                if self.current_game_id == game_id:
                    self.current_game_id = None
                return

            initial_state = event.get('state', {})
            moves_str = initial_state.get('moves', '')
            status = initial_state.get('status')
            print(f"gameFull state: Status='{status}', Moves='{moves_str}'")

            if self.color == chess.WHITE and not moves_str and status == 'started':
                print(f"Game {game_id} starts, bot is WHITE. Making first move.")
                board = self._reconstruct_board(moves_str)
                if board.fen() == chess.STARTING_FEN:
                    self.make_move(game_id, chess.STARTING_FEN)
                else:
                    print(f"Warning: Making first move as White, but board FEN is not standard start: {board.fen()}")
                    self.make_move(game_id, board.fen())

        except KeyError as e:
            print(f"KeyError in _handle_game_full_event for game {game_id}: {e}")
            print(f"Event data: {event}")
            if self.current_game_id == game_id: self.current_game_id = None
        except Exception as e:
            print(f"Unexpected error in _handle_game_full_event for game {game_id}: {e}")
            print(f"Event data: {event}")
            if self.current_game_id == game_id: self.current_game_id = None

    def _handle_game_state_event(self, event: dict, game_id: str):
        if self.color is None:
            print(f"Error: self.color is None in _handle_game_state_event for game {game_id}. Cannot process.")
            return
        try:
            moves_str = event.get('moves', '')
            status = event.get('status')
            if status not in ['created', 'started']:
                return

            board = self._reconstruct_board(moves_str)
            if board.is_game_over():
                return

            if board.turn == self.color:
                self.make_move(game_id, board.fen())
            else:
                pass

        except Exception as e:
            print(f"Error processing gameState for game {game_id}: {e}")
            print(f"Event data: {event}")
            if self.current_game_id == game_id: self.current_game_id = None

    def _reconstruct_board(self, moves_str: str) -> chess.Board:
        board = chess.Board()
        if moves_str:
            uci_moves = moves_str.split()
            for move in uci_moves:
                if not move: continue
                try:
                    board.push_uci(move)
                except ValueError as e:
                    print(
                        f"ERROR: Could not parse UCI move '{move}' from sequence '{moves_str}'. Board state may be incorrect! Error: {e}")
                    pass
        return board

    def make_move(self, game_id: str, fen: str):
        if self.current_game_id != game_id:
            print(f"Attempted to make move in {game_id}, but current game is {self.current_game_id}. Aborting move.")
            return

        print(f"Calculating move for FEN: {fen} in game {game_id}")
        move = None
        try:
            board = chess.Board(fen)
            if board.is_game_over():
                print(f"Board is already game over ({board.result()}) in make_move for {game_id}. No move needed.")
                return
            if not board.legal_moves:
                print(f"No legal moves available on board for FEN {fen} in game {game_id}.")
                return

            move = self.engine.get_best_move(fen)

            if move is None:
                print(f"Engine {type(self.engine).__name__} returned no move for game {game_id}, FEN: {fen}.")
                return

            move_uci = move.uci()
            print(f"Engine selected move: {move_uci} for game {game_id}")

            self.client.bots.make_move(game_id, move_uci)

        except berserk.exceptions.ResponseError as e:
            move_repr = move.uci() if move else 'N/A'
            print(f"API Error making move {move_repr} in game {game_id}: {e.status_code} {e.error}")
            print(f"Lichess error details: {e.error}")
            if self.current_game_id == game_id: self.current_game_id = None
        except Exception as e:
            move_repr = move.uci() if move else 'N/A'
            print(f"Unexpected error making move {move_repr} in game {game_id}: {e}")
            # Stop playing this game on unexpected error
            if self.current_game_id == game_id: self.current_game_id = None


def main():
    parser = argparse.ArgumentParser(description='Run a Lichess bot using a CNN model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the CNN model file')
    parser.add_argument("--token", type=str, required=True, help="Lichess API token")

    args = parser.parse_args()
    api_token = args.token

    model_path = args.model

    print(f"Initializing bot with model: {model_path}")
    try:
        bot = LichessBot(api_token, model_path)
    except ValueError as e:
        print(f"Error initializing bot: {e}")
        return
    except Exception as e:
        print(f"Unexpected error initializing bot: {e}")
        return

    while True:
        try:
            bot.handle_game_stream()
            print("Event stream handler stopped. Restarting in 60 seconds...")
            time.sleep(60)
        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except RuntimeError as e:
            print(f"Runtime Error: {e}. Stopping bot.")
            break

if __name__ == "__main__":
    main()
