# physical_board_detector.py
import subprocess
import os
import tempfile
import shutil
import time
import numpy as np
import chess # Now needed for Board object type hint and validation

# Import necessary functions from the updated vision module
try:
    # Now need the integer state array function
    from image_to_fen import (
        get_board_state_array_from_image_path, # <-- Changed from get_occupancy...
        map_array_indices_to_chess_square,
        APP_TEMP_PARENT_DIR
    )
    VISION_FUNCTIONS_LOADED = True
except ImportError as e:
    print(f"[CRITICAL ERROR] Could not import required vision functions from image_to_fen.py: {e}")
    APP_TEMP_PARENT_DIR = None
    VISION_FUNCTIONS_LOADED = False
    def get_board_state_array_from_image_path(path): return None # Dummy
    def map_array_indices_to_chess_square(r, f): return None

CAMERA_DIR = "/sdcard/DCIM/Camera" # Standard Android Camera path
ADB_COMMAND = "adb"

# --- ADB Utilities (Unchanged from previous version) ---
def _run_adb_command(cmd_args: list, timeout_sec=15, suppress_errors=False) -> tuple[bool, str]:
    """Runs an ADB command, returning (success_bool, output_str/error_str)."""
    try:
        full_command = [ADB_COMMAND] + cmd_args
        result = subprocess.run(full_command, capture_output=True, text=True, check=False, timeout=timeout_sec, errors="ignore")
        if result.returncode == 0: return True, result.stdout.strip()
        else:
            error_message = result.stderr.strip() if result.stderr else result.stdout.strip()
            if not suppress_errors:
                 if "No such file or directory" not in error_message: # Avoid spamming for ls -t on empty dir
                    print(f"[ADB ERROR] Command failed: {' '.join(full_command)}\n            Output: {error_message}")
            if "device not found" in error_message or "no devices/emulators found" in error_message: raise RuntimeError("ADB device not found. Check connection, USB debugging, and PC trust.")
            elif "File not found" in error_message or "No such file or directory" in error_message: raise FileNotFoundError(f"ADB Error: Path specified in command likely incorrect ('{cmd_args[-1]}'?).")
            elif "device unauthorized" in error_message: raise PermissionError("ADB device found but is unauthorized. Check phone screen.")
            elif "offline" in error_message: raise ConnectionAbortedError("ADB device found but is offline. Try reconnecting/restarting.")
            return False, error_message
    except FileNotFoundError: raise RuntimeError(f"ADB command '{ADB_COMMAND}' not found. Ensure ADB is installed and in your system's PATH.")
    except subprocess.TimeoutExpired: print(f"[ADB ERROR] Command timed out after {timeout_sec}s: {' '.join(full_command)}"); return False, f"Command timed out ({timeout_sec}s)."
    except (RuntimeError, PermissionError, ConnectionAbortedError, FileNotFoundError) as e: raise
    except Exception as e: print(f"[ADB ERROR] Unexpected error running command: {' '.join(full_command)}\n            {e}"); return False, f"Unexpected ADB error: {e}"

def get_newest_photo_from_phone() -> str | None:
    """Finds the newest JPG/JPEG photo name on the phone via ADB."""
    print(f"[INFO] Searching for newest photo in '{CAMERA_DIR}' on phone...")
    try:
        success, output = _run_adb_command(["shell", "ls", "-t", CAMERA_DIR])
        if not success:
            if "Path specified in command likely incorrect" in output: print(f"[ERROR] Critical: Camera directory '{CAMERA_DIR}' not found on device."); raise FileNotFoundError(f"Camera directory '{CAMERA_DIR}' not found.")
            else: raise RuntimeError(f"Failed to list files in {CAMERA_DIR}. ADB error: {output}")
        for name in output.splitlines():
            name = name.strip()
            if name and name.lower().endswith((".jpg", ".jpeg")) and not name.startswith('.'): print(f"[INFO] Newest photo found on device: '{name}'"); return name
        print(f"[WARN] No JPG or JPEG files found in {CAMERA_DIR} on the phone."); return None
    except FileNotFoundError as e: raise RuntimeError(f"Cannot list photos: {e}") # Re-raise specific errors
    except RuntimeError as e: raise
    except Exception as e: raise RuntimeError(f"Unexpected error finding newest photo: {e}")


def pull_photo_from_phone(remote_filename: str, local_dir: str) -> str | None:
    """Pulls a photo from the phone to a local directory."""
    if not remote_filename: print("[ERROR] Remote filename is empty."); return None
    if not local_dir: print("[ERROR] Local directory is empty."); return None
    try:
        safe_remote_filename = os.path.basename(remote_filename) # Sanitize
        remote_path = f"{CAMERA_DIR}/{safe_remote_filename}"
        local_path = os.path.join(local_dir, safe_remote_filename)
        os.makedirs(local_dir, exist_ok=True) # Ensure local_dir exists
        print(f"[INFO] Pulling '{remote_path}' to '{local_path}'...")
        success, output = _run_adb_command(["pull", remote_path, local_path], timeout_sec=30) # Increased timeout for large files
        if success:
             if os.path.exists(local_path) and os.path.getsize(local_path) > 0: print(f"[INFO] Photo pulled successfully: {local_path}"); return local_path
             else: print(f"[ERROR] ADB pull reported success, but local file '{local_path}' is missing or empty."); return None
        else:
            if "does not exist" in output or "No such file or directory" in output: print(f"[ERROR] ADB pull failed: Remote file '{remote_path}' likely does not exist.") # More specific error
            print(f"[ERROR] ADB pull command failed. ADB Message: {output}"); return None
    except Exception as e: print(f"[ERROR] An unexpected error occurred while pulling photo: {e}"); return None


def check_adb_device() -> bool:
    """Checks for at least one authorized ADB device."""
    print("[INFO] Checking for ADB device connection and authorization...")
    try:
        success, output = _run_adb_command(["devices"], timeout_sec=5, suppress_errors=True) # Suppress initial benign errors
        if not success and "ADB command" in output : print(f"[ERROR] {output}"); return False # ADB not found
        if not success: print(f"[ERROR] Failed to run 'adb devices'. Error: {output}"); return False

        lines = output.strip().splitlines(); device_found, unauthorized_found, offline_found = False, False, False
        if len(lines) <= 1: print("[ERROR] No ADB devices found."); return False # "List of devices attached" is line 1

        for line in lines[1:]: # Skip header line
            line = line.strip(); parts = line.split()
            if len(parts) == 2:
                status = parts[1].lower()
                if status == "device": print(f"[INFO] Found authorized device: {parts[0]}"); device_found = True
                elif status == "unauthorized": print(f"[ERROR] Found unauthorized device: {parts[0]}. Check phone screen."); unauthorized_found = True
                elif status == "offline": print(f"[ERROR] Found offline device: {parts[0]}. Try reconnecting."); offline_found = True
                else: print(f"[WARN] Found device with unknown status: {line}")

        if device_found: return True
        elif unauthorized_found or offline_found: return False # Specific error already printed
        else: print("[ERROR] No authorized and online ADB devices found."); return False
    except RuntimeError as e: print(f"[ERROR] {e}"); return False # From _run_adb_command for ADB not found
    except Exception as e: print(f"[ERROR] Unexpected error checking ADB devices: {e}"); return False

# --- Move Detection Logic ---

def find_move_from_state_diff(before_state_array: np.ndarray, after_state_array: np.ndarray, board: chess.Board) -> str | None:
    """
    Compares two 8x8 integer state arrays (0, +1, -1) and uses the logical board
    state to deduce the most likely legal move made.
    Returns the move in UCI format (e.g., "e2e4") or None if unclear.
    """
    if before_state_array is None or after_state_array is None:
        print("[ERROR] find_move_from_state_diff: Input arrays cannot be None.")
        return None
    if not isinstance(before_state_array, np.ndarray) or before_state_array.shape != (8,8):
         before_state_array = np.array(before_state_array) # Attempt conversion
         if before_state_array.shape != (8,8):
              print("[ERROR] find_move_from_state_diff: Invalid 'before' state array shape.")
              return None
    if not isinstance(after_state_array, np.ndarray) or after_state_array.shape != (8,8):
         after_state_array = np.array(after_state_array) # Attempt conversion
         if after_state_array.shape != (8,8):
              print("[ERROR] find_move_from_state_diff: Invalid 'after' state array shape.")
              return None

    diff = after_state_array - before_state_array
    turn_color = board.turn # True for White (expects +1), False for Black (expects -1)

    potential_origins = []
    potential_destinations = []

    # Define expected diff values based on whose turn it is
    departure_val = -1 if turn_color == chess.WHITE else +1 # Piece of current player's color disappears from origin
    arrival_val_move = +1 if turn_color == chess.WHITE else -1 # Piece of current player's color appears on destination (simple move)
    # If white (1) moves to a square previously occupied by black (-1), diff = 1 - (-1) = 2
    # If black (-1) moves to a square previously occupied by white (1), diff = -1 - 1 = -2
    arrival_val_capture_by_white = 2  # White captures black: 1 (white) - (-1) (black) = 2
    arrival_val_capture_by_black = -2 # Black captures white: -1 (black) - 1 (white) = -2


    # Find indices based on diff values
    for r in range(8):
        for f in range(8):
            d = diff[r, f]
            sq_name = map_array_indices_to_chess_square(r, f)
            if not sq_name: continue # Skip if mapping fails

            if d == departure_val: # Square became empty from perspective of current player
                potential_origins.append(sq_name)
            elif turn_color == chess.WHITE and (d == arrival_val_move or d == arrival_val_capture_by_white):
                potential_destinations.append(sq_name)
            elif turn_color == chess.BLACK and (d == arrival_val_move or d == arrival_val_capture_by_black):
                potential_destinations.append(sq_name)
            elif d != 0: # Some other unexpected change
                 # This might be due to en-passant where a third square changes, or vision error
                 # For en-passant: if white moves e2e4, then black d7d5, then white e4d5ep.
                 # 'd5' (to_sq for white) changes from -1 (black pawn) to 1 (white pawn) -> diff = 2
                 # 'e4' (from_sq for white) changes from 1 (white pawn) to 0 -> diff = -1
                 # 'd4' (captured black pawn) changes from -1 (black pawn) to 0 -> diff = +1 (unexpected here)
                 # This simple diff logic might struggle with en passant without special handling.
                 # The current logic relies on board.legal_moves to resolve this.
                 print(f"[DEBUG] find_move_from_state_diff: Unexpected diff value {d} at {sq_name} for {'White' if turn_color else 'Black'} to move.")


    # --- Disambiguation using legal moves ---
    possible_moves_uci = []
    try:
        all_legal_moves = list(board.legal_moves) # Generate once
    except Exception as e:
        print(f"[ERROR] Could not generate legal moves from board state: {e}")
        print(f"        Board FEN: {board.fen()}")
        return None # Cannot proceed without legal moves

    if not all_legal_moves:
        print("[WARN] No legal moves available in the current position.")
        return None

    for origin_sq_name in potential_origins:
        try:
             origin_sq_idx = chess.parse_square(origin_sq_name)
        except ValueError: continue

        for move in all_legal_moves:
            if move.from_square == origin_sq_idx:
                dest_sq_name_from_move = chess.square_name(move.to_square)
                if dest_sq_name_from_move in potential_destinations:
                    possible_moves_uci.append(move.uci())
        
    if not possible_moves_uci: # If simple check failed, consider castling more directly
        for move in all_legal_moves:
            if board.is_castling(move):
                king_from_sq_name = chess.square_name(move.from_square)
                king_to_sq_name = chess.square_name(move.to_square)
                if king_from_sq_name in potential_origins and king_to_sq_name in potential_destinations:
                    # Check if rook squares also changed appropriately (this is harder with simple diff)
                    # For now, trust that if king move matches and it's a legal castle, it's likely.
                    possible_moves_uci.append(move.uci())
                    print(f"[DEBUG] Castling move {move.uci()} considered based on king's movement match.")


    possible_moves_uci = sorted(list(set(possible_moves_uci))) # Remove duplicates

    if len(possible_moves_uci) == 1:
        detected_move = possible_moves_uci[0]
        print(f"[INFO] Unique legal move found matching state diff: {detected_move}")
        return detected_move
    elif len(possible_moves_uci) == 0:
        print(f"[ERROR] No legal moves found matching the state difference.")
        print(f"        Potential Origins (squares that became empty for current player): {potential_origins}")
        print(f"        Potential Destinations (squares that became occupied by current player or captured opponent): {potential_destinations}")
        print(f"        Difference Array (After - Before):\n{diff}")
        print(f"        Board FEN before move: {board.fen()}")
        print(f"        Legal moves were: {[m.uci() for m in all_legal_moves]}")
        return None
    else: # Multiple legal moves match
        print(f"[WARN] Ambiguous move: Multiple legal moves match state difference: {possible_moves_uci}")
        print(f"        Potential Origins: {potential_origins}, Potential Destinations: {potential_destinations}")
        return None


def get_latest_board_state_from_phone() -> np.ndarray | None:
    """
    Fetches the latest image via ADB, processes it, returns the 8x8 integer
    state array (0, +1, -1). Returns None on failure.
    """
    if not VISION_FUNCTIONS_LOADED: print("[ERROR] Vision functions not loaded."); return None
    if not APP_TEMP_PARENT_DIR: print("[ERROR] Vision temporary directory not configured."); return None

    local_image_path = None
    int_state_array = None
    temp_dir_session = None

    try:
        remote_file = get_newest_photo_from_phone()
        if not remote_file: return None # Error already printed by get_newest_photo_from_phone

        temp_dir_session = tempfile.mkdtemp(prefix="board_capture_", dir=APP_TEMP_PARENT_DIR)
        local_image_path = pull_photo_from_phone(remote_file, temp_dir_session)

        if local_image_path and os.path.exists(local_image_path):
            print(f"[INFO] Processing image '{os.path.basename(local_image_path)}' for board state array...")
            int_state_array = get_board_state_array_from_image_path(local_image_path)
            if int_state_array is None:
                 print("[ERROR] Failed to get board state array from image processing.")
                 return None # Error from get_board_state_array_from_image_path
            else:
                 white_count = np.count_nonzero(int_state_array == 1)
                 black_count = np.count_nonzero(int_state_array == -1)
                 if white_count + black_count < 2 or white_count + black_count > 32: # Basic sanity
                      print(f"[WARN] State array generated, but piece count seems unusual (W:{white_count}, B:{black_count}).")
                 return int_state_array
        else:
            print(f"[ERROR] Failed to retrieve or locate image file locally ('{local_image_path}').")
            return None

    except (RuntimeError, ValueError, FileNotFoundError, TimeoutError, ConnectionAbortedError, PermissionError) as e: # Catch specific errors from ADB/file ops
        print(f"[ERROR] Failed board state capture due to ADB/File error: {e}")
        return None
    except Exception as e: # Catch any other unexpected errors
        print(f"[ERROR] Unexpected error during board state capture: {e}")
        return None
    finally:
        if temp_dir_session and os.path.exists(temp_dir_session):
            try: shutil.rmtree(temp_dir_session)
            except OSError as e_rm: print(f"[WARN] Could not delete temp dir {temp_dir_session}: {e_rm}")


# Updated signature to include the board object
def detect_human_move(previous_board_state_array: np.ndarray | None, board: chess.Board) -> tuple[str | None, np.ndarray | None]:
    """
    Gets current board state array via photo (after user is prompted by game_logic),
    compares with previous, uses board.legal_moves for disambiguation.
    Returns (detected_move_uci, new_board_state_array) or (None, None).
    --- Note: The input() prompt was removed from here and is now handled by game_logic.py ---
    """
    if previous_board_state_array is None:
        print("[ERROR] Cannot detect move: Invalid previous board state provided to detect_human_move.")
        return None, None

    # The input() prompt to the user has been moved to game_logic.py's game_loop_physical.
    # This function now directly proceeds to get and analyze the board state.

    print("[INFO] Getting current board state from phone (called by detect_human_move)...")
    current_board_state_array = get_latest_board_state_from_phone()

    if current_board_state_array is None:
        print("[ERROR] Could not determine current board state from photo (detect_human_move).")
        return None, None # Indicate failure to get state

    print("[INFO] Analyzing move based on state difference and legal moves (detect_human_move)...")
    detected_move_uci = find_move_from_state_diff(previous_board_state_array, current_board_state_array, board)

    if detected_move_uci:
        print("-" * 15)
        print(f" >>> Detected Move (by detect_human_move): {detected_move_uci} <<<")
        print("-" * 15)
        return detected_move_uci, current_board_state_array
    else:
        print("[ERROR] Could not reliably detect a valid move from the board state change (detect_human_move).")
        return None, None