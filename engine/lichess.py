import argparse
import time
from typing import Optional

import berserk
import chess

from engine.cnn_bot import CNNChessBot


class LichessBot:
    def __init__(self, api_token: str, model_path: str):
        """Initialize the Lichess bot with the provided API token and model path."""
        if not model_path:
            raise ValueError("A valid model path must be provided.")
        self.session = berserk.TokenSession(api_token)
        self.client = berserk.Client(self.session)
        self.current_game_id: Optional[str] = None
        self.color: Optional[chess.Color] = None
        self.engine = CNNChessBot(model_path)
        self.bot_id: Optional[str] = None

    def fetch_bot_id(self):
        """Fetch and store the bot's account ID. Returns True on success, False otherwise."""
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
        """Main loop to handle incoming game events and challenges."""
        if not self.fetch_bot_id():
            print("Failed to fetch bot ID. Cannot proceed.")
            # Depending on desired behavior, you might retry or exit.
            # Raising an exception might be appropriate here.
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

                        # Example: Accept only standard rated correspondence games
                        variant = challenge.get('variant', {}).get('key')
                        speed = challenge.get('speed')
                        rated = challenge.get('rated', False)

                        # Customize your acceptance criteria here
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
                                # Optionally, you could try to abort/resign the existing game first.
                                continue

                            print(f"Starting game {game_id}")
                            # Reset game-specific state
                            self.current_game_id = game_id
                            self.color = None  # Reset color, will be determined by gameFull
                            # Start processing this specific game in a separate thread/task?
                            # For simplicity here, we call it directly, but this blocks handling other events.
                            # A better architecture would use asyncio or threading.
                            self.play_game(game_id)
                            # If play_game runs synchronously until the game ends,
                            # current_game_id might be None here.
                            print(f"Finished processing game {game_id} in handle_game_stream.")

                        else:
                            print("Error: gameStart event received without game ID.")

                    elif event['type'] == 'gameFinish':
                        game_info = event.get('game', {})
                        game_id = game_info.get('id')
                        if game_id:
                            print(f"Game {game_id} finished event received.")
                            # If this was the game being played, clear the state
                            if self.current_game_id == game_id:
                                print(f"Clearing state for finished game {game_id}.")
                                self.current_game_id = None
                                self.color = None
                        else:
                            print("Error: gameFinish event received without game ID.")

            except berserk.exceptions.ResponseError as e:
                print(f"API Response Error in event stream: {e.status_code} {e.error}")
                if e.status_code == 401:  # Unauthorized - likely bad token
                    print("Authentication failed. Check API token. Stopping event stream.")
                    raise  # Re-raise to stop the main loop or handle appropriately
                print("Waiting 10 seconds before retrying event stream...")
                time.sleep(10)  # Wait longer for API errors
            except Exception as e:
                print(f"Unexpected error in event stream: {e}")
                print("Waiting 5 seconds before retrying event stream...")
                time.sleep(5)  # Wait before potentially retrying connection

    def play_game(self, game_id: str):
        """Manage gameplay for a single game by streaming its state."""
        print(f"Streaming state for game {game_id}...")
        try:
            # Ensure bot ID is available (should be fetched already, but double-check)
            if not self.bot_id:
                print(f"Cannot play game {game_id}: Bot ID not available.")
                return  # Exit this game processing

            stream = self.client.bots.stream_game_state(game_id)
            for event in stream:
                # Make sure we are still supposed to be playing *this* game
                if self.current_game_id != game_id:
                    print(f"Game context switched away from {game_id} (now {self.current_game_id}). Stopping stream.")
                    break  # Stop processing this old game stream

                event_type = event.get('type')

                if event_type == 'gameFull':
                    # Process the initial full game state
                    self._handle_game_full_event(event, game_id)

                elif event_type == 'gameState':
                    # Process incremental game state updates
                    # Check if color has been determined (should happen in gameFull)
                    if self.color is None:
                        print(f"Warning: Received gameState for {game_id} before color was determined. Ignoring state.")
                        # This indicates an issue, maybe gameFull was missed or processed incorrectly.
                        # You might need logic here to request the full state again.
                        continue

                    self._handle_game_state_event(event, game_id)

                    # Check status from gameState to see if game ended
                    status = event.get('status')
                    if status not in ['created', 'started']:
                        print(f"Game {game_id} status changed to '{status}' in gameState. Ending game processing.")
                        break  # Exit the loop for this game stream

                elif event_type == 'chatLine':
                    # Optional: Handle chat messages
                    pass
                elif event_type == 'opponentGone':
                    # Optional: Handle opponent disconnection (may need timer logic)
                    pass

        except berserk.exceptions.ResponseError as e:
            print(f"API Response Error in game stream {game_id}: {e.status_code} {e.error}")
            # Handle specific errors e.g., 404 Not Found if game ID is invalid
            if e.status_code == 404:
                print(f"Game {game_id} not found.")
        except Exception as e:
            print(f"Unexpected error processing game stream for {game_id}: {e}")
        finally:
            # Ensure game state is cleared if this was the active game and it ended/errored out
            if self.current_game_id == game_id:
                print(f"Cleaning up state for game {game_id} after stream ended/errored.")
                self.current_game_id = None
                self.color = None

    def _handle_game_full_event(self, event: dict, game_id: str):
        """Handle initial game state when a game starts."""
        print(f"Handling gameFull for {game_id}.")
        try:
            # Ensure bot ID is available
            if not self.bot_id:
                print(f"Error in _handle_game_full_event for {game_id}: Bot ID not available.")
                return

            # Use .get() for safer access to player IDs
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
                self.color = None  # Indicate color couldn't be determined
                # Stop processing this game if color assignment fails
                if self.current_game_id == game_id:
                    self.current_game_id = None
                return

            # Get initial state details safely
            initial_state = event.get('state', {})
            moves_str = initial_state.get('moves', '')
            status = initial_state.get('status')
            print(f"gameFull state: Status='{status}', Moves='{moves_str}'")

            # Check if it's the bot's turn to move immediately (White's turn, no moves made yet, game started)
            if self.color == chess.WHITE and not moves_str and status == 'started':
                print(f"Game {game_id} starts, bot is WHITE. Making first move.")
                # Reconstruct board for FEN (should be starting FEN)
                board = self._reconstruct_board(moves_str)
                if board.fen() == chess.STARTING_FEN:
                    self.make_move(game_id, chess.STARTING_FEN)
                else:
                    # Should not happen if moves_str is empty, but good sanity check
                    print(f"Warning: Making first move as White, but board FEN is not standard start: {board.fen()}")
                    self.make_move(game_id, board.fen())

            # If Black, or if White but moves already exist/status not 'started', wait for gameState update

        except KeyError as e:
            print(f"KeyError in _handle_game_full_event for game {game_id}: {e}")
            print(f"Event data: {event}")
            if self.current_game_id == game_id: self.current_game_id = None  # Stop game on error
        except Exception as e:
            print(f"Unexpected error in _handle_game_full_event for game {game_id}: {e}")
            print(f"Event data: {event}")
            if self.current_game_id == game_id: self.current_game_id = None  # Stop game on error

    def _handle_game_state_event(self, event: dict, game_id: str):
        """Handle ongoing game state updates."""
        # --- Debug: Entered function ---
        # print(f"DEBUG: _handle_game_state_event entered for game {game_id}.")

        # Color should have been set by gameFull
        if self.color is None:
            print(f"Error: self.color is None in _handle_game_state_event for game {game_id}. Cannot process.")
            return

        # --- Debug: Print current color ---
        # print(f"DEBUG: Current bot color: {'WHITE' if self.color == chess.WHITE else 'BLACK'}")

        try:
            moves_str = event.get('moves', '')
            status = event.get('status')

            # --- Debug: Print received state ---
            # print(f"DEBUG: Received gameState - Status: '{status}', Moves: '{moves_str}'")

            if status not in ['created', 'started']:
                # print(f"DEBUG: Game {game_id} status is '{status}'. Not attempting move.")
                # The main loop in play_game should catch this and break the stream.
                return

            board = self._reconstruct_board(moves_str)
            # --- Debug: Print board turn ---
            # print(f"DEBUG: Reconstructed board. Board turn: {'WHITE' if board.turn == chess.WHITE else 'BLACK'}")

            # Check if the game is over according to the board state
            if board.is_game_over():
                # print(f"DEBUG: Board indicates game is over ({board.result()}). No move to make.")
                # The main loop should detect the status change and terminate soon.
                return

            # Check whose turn it is *and* if it matches the bot's color
            if board.turn == self.color:
                # --- Debug: Turn condition met ---
                # print(f"DEBUG: Board turn ({'WHITE' if board.turn == chess.WHITE else 'BLACK'}) matches bot color. Calling make_move.")
                self.make_move(game_id, board.fen())
            else:
                # --- Debug: Turn condition NOT met ---
                # print(f"DEBUG: Board turn ({'WHITE' if board.turn == chess.WHITE else 'BLACK'}) does NOT match bot color ({'WHITE' if self.color == chess.WHITE else 'BLACK'}). Waiting.")
                pass  # It's the opponent's turn, do nothing.

        except Exception as e:
            print(f"Error processing gameState for game {game_id}: {e}")
            print(f"Event data: {event}")
            # Consider if game should be stopped on gameState error
            if self.current_game_id == game_id: self.current_game_id = None

    def _reconstruct_board(self, moves_str: str) -> chess.Board:
        """Reconstruct a chess board from a Lichess moves string (space-separated UCI)."""
        board = chess.Board()
        if moves_str:
            uci_moves = moves_str.split()
            for move in uci_moves:
                if not move: continue  # Skip empty strings if moves_str ends with space
                try:
                    board.push_uci(move)
                except ValueError as e:
                    # Can happen with illegal moves in string, promotion formats etc.
                    print(
                        f"ERROR: Could not parse UCI move '{move}' from sequence '{moves_str}'. Board state may be incorrect! Error: {e}")
                    # Depending on severity, you might want to stop processing the game.
                    # For now, we continue, acknowledging the risk.
                    pass
        return board

    def make_move(self, game_id: str, fen: str):
        """Calculate and execute the best move for the current position."""
        # Ensure this is still the active game
        if self.current_game_id != game_id:
            print(f"Attempted to make move in {game_id}, but current game is {self.current_game_id}. Aborting move.")
            return

        print(f"Calculating move for FEN: {fen} in game {game_id}")
        move = None  # Initialize move to None
        try:
            # Check board state before calling engine
            board = chess.Board(fen)
            if board.is_game_over():
                print(f"Board is already game over ({board.result()}) in make_move for {game_id}. No move needed.")
                return
            if not board.legal_moves:
                print(f"No legal moves available on board for FEN {fen} in game {game_id}.")
                # Consider resigning or offering draw?
                # self.client.bots.resign_game(game_id) ?
                return

            # Get move from engine
            move = self.engine.get_best_move(fen)  # Your CNNChessBot call

            if move is None:
                print(f"Engine {type(self.engine).__name__} returned no move for game {game_id}, FEN: {fen}.")
                # Fallback? Resign? For now, just log and return.
                return

            move_uci = move.uci()
            print(f"Engine selected move: {move_uci} for game {game_id}")

            # Ensure the move is legal according to python-chess as a sanity check
            try:
                # This will raise ValueError if the move is illegal in the current position
                board.find_move(move.from_square, move.to_square, move.promotion)
            except ValueError:
                print(f"CRITICAL ERROR: Engine proposed illegal move {move_uci} for FEN {fen} in game {game_id}.")
                # Do not send the illegal move.
                # Consider resigning or alternative action.
                # self.client.bots.resign_game(game_id)
                if self.current_game_id == game_id: self.current_game_id = None  # Stop playing
                return

            # Make the move via Lichess API
            self.client.bots.make_move(game_id, move_uci)
            print(f"Successfully sent move {move_uci} for game {game_id}")

        except berserk.exceptions.ResponseError as e:
            move_repr = move.uci() if move else 'N/A'
            print(f"API Error making move {move_repr} in game {game_id}: {e.status_code} {e.error}")
            # If it's 400 Bad Request, check the error message from Lichess if available in e.error
            # Common reasons: {"error":"Not your turn, or game already over"} or {"error":"Invalid move format"} or {"error":"Illegal move"}
            print(f"Lichess error details: {e.error}")
            # Stop playing this game if the API rejects our move
            if self.current_game_id == game_id: self.current_game_id = None
        except Exception as e:
            move_repr = move.uci() if move else 'N/A'
            print(f"Unexpected error making move {move_repr} in game {game_id}: {e}")
            # Stop playing this game on unexpected error
            if self.current_game_id == game_id: self.current_game_id = None


def main():
    """Entry point to start the bot."""

    parser = argparse.ArgumentParser(description='Run a Lichess bot using a CNN model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the CNN model file')
    parser.add_argument("--token", type=str, required=True, help="Lichess API token")
    # Add option for API token for better security?
    # parser.add_argument('--token', type=str, help='Lichess API token (or set LICHESS_API_TOKEN env var)')

    args = parser.parse_args()
    api_token = args.token

    # Use model path from arguments
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
            # Start the main event loop. This will run indefinitely until an error forces it to exit.
            # It internally handles fetching the bot ID and streaming events.
            bot.handle_game_stream()
            # If handle_game_stream returns (e.g., due to auth error re-raised), wait before potentially restarting
            print("Event stream handler stopped. Restarting in 60 seconds...")
            time.sleep(60)
        except KeyboardInterrupt:
            print("Bot stopped by user (KeyboardInterrupt).")
            break
        except RuntimeError as e:
            # Catch specific errors like failing to get bot ID
            print(f"Runtime Error: {e}. Stopping bot.")
            break
        except Exception as e:
            # Catch any other unexpected exceptions that might crash the main loop
            print(f"Critical error in main loop: {type(e).__name__}: {e}. Restarting in 30 seconds...")
            time.sleep(30)

    print("Bot exited.")


if __name__ == "__main__":
    main()
