# game_logic.py
import pygame
import chess
import chess.engine
import time
import os
import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np

# --- Try to import detection and vision modules ---
try:
    from physical_board_detector import (
        detect_human_move,
        check_adb_device
    )
    from image_to_fen import (
         map_chess_square_to_array_indices,
         get_fen_from_image_path # Used by UI
    )
    PHONE_DETECTION_ENABLED = True
    print("[INFO] Physical board detector and image_to_fen modules loaded successfully.")
except ImportError as e:
    print(f"[WARN] Could not import physical_board_detector or image_to_fen: {e}")
    print("[WARN] Physical board detection will be disabled.")
    PHONE_DETECTION_ENABLED = False
    # Dummy functions if imports fail, to allow basic script structure to be parsed
    def check_adb_device(): return False
    def detect_human_move(arr, board): # arr is current_board_state_array, board is chess.Board
        print("[WARN-DUMMY] detect_human_move called, but module not loaded.")
        return None, None
    def map_chess_square_to_array_indices(sq): return None, None
    def get_fen_from_image_path(path): return None

# --- IMPORTANT: SET YOUR STOCKFISH PATH HERE ---
STOCKFISH_PATH = r"" # <<<--- UPDATE THIS PATH

# --- Constants ---
SQUARE_SIZE = 75
BOARD_WIDTH = 8 * SQUARE_SIZE
BOARD_HEIGHT = 8 * SQUARE_SIZE
INFO_PANEL_HEIGHT = 80  # Ensure this is adequate for multi-line messages
SCREEN_WIDTH = BOARD_WIDTH
SCREEN_HEIGHT = BOARD_HEIGHT + INFO_PANEL_HEIGHT
LIGHT_SQUARE_COLOR = (238, 238, 210)
DARK_SQUARE_COLOR = (118, 150, 86)
BACKGROUND_COLOR = (40, 40, 40)
TEXT_COLOR = (220, 220, 220)
PIECE_COLOR_WHITE = (200, 200, 200)
PIECE_COLOR_BLACK = (25, 25, 25)
OUTLINE_COLOR_FOR_WHITE_PIECES = (0, 0, 0)
OUTLINE_THICKNESS = 1
UNICODE_PIECES = {'P':'♙','R':'♖','N':'♘','B':'♗','Q':'♕','K':'♔','p':'♟','r':'♜','n':'♞','b':'♝','q':'♛','k':'♚'}
HUMAN_PLAYER_COLOR = chess.WHITE
ENGINE_PLAYER_COLOR = chess.BLACK

# --- Pygame Drawing Functions ---
def draw_board_and_pieces(screen, board, piece_font):
    for r_disp in range(8): # 0 to 7
        for f_disp in range(8): # 0 to 7
            # chess.square(file_index, rank_index)
            # Display rank 8 (board__disp_r=0) is chess rank 7. Display rank 1 (board_disp_r=7) is chess rank 0.
            square_index = chess.square(f_disp, 7 - r_disp) # Correct mapping
            rect = pygame.Rect(f_disp * SQUARE_SIZE, r_disp * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            color = LIGHT_SQUARE_COLOR if (r_disp + f_disp) % 2 == 0 else DARK_SQUARE_COLOR
            pygame.draw.rect(screen, color, rect)

            piece = board.piece_at(square_index)
            if piece:
                piece_char = UNICODE_PIECES.get(piece.symbol(), '?')
                main_piece_color = PIECE_COLOR_WHITE if piece.color == chess.WHITE else PIECE_COLOR_BLACK
                
                # Draw outline for white pieces for better visibility on light squares
                if piece.color == chess.WHITE:
                    offsets = [(-OUTLINE_THICKNESS, -OUTLINE_THICKNESS), (OUTLINE_THICKNESS, -OUTLINE_THICKNESS),
                               (-OUTLINE_THICKNESS, OUTLINE_THICKNESS), (OUTLINE_THICKNESS, OUTLINE_THICKNESS),
                               (0, -OUTLINE_THICKNESS), (0, OUTLINE_THICKNESS),
                               (-OUTLINE_THICKNESS, 0), (OUTLINE_THICKNESS, 0)]
                    for dx, dy in offsets:
                        outline_surface = piece_font.render(piece_char, True, OUTLINE_COLOR_FOR_WHITE_PIECES)
                        outline_rect = outline_surface.get_rect(center=rect.center)
                        screen.blit(outline_surface, outline_rect.move(dx, dy))
                
                text_surface = piece_font.render(piece_char, True, main_piece_color)
                text_rect = text_surface.get_rect(center=rect.center)
                screen.blit(text_surface, text_rect)

def update_display(screen, board, piece_font, info_font, status_text):
    screen.fill(BACKGROUND_COLOR)
    draw_board_and_pieces(screen, board, piece_font)
    
    # Info Panel Background
    info_panel_rect = pygame.Rect(0, BOARD_HEIGHT, SCREEN_WIDTH, INFO_PANEL_HEIGHT)
    pygame.draw.rect(screen, BACKGROUND_COLOR, info_panel_rect)

    # Add "Check!" to status if applicable
    status_to_display = status_text
    if board and board.is_check() and not board.is_game_over(claim_draw=True): # Check if board is not None
        status_to_display += " (Check!)"

    # Text Wrapping for Info Panel
    max_width = SCREEN_WIDTH - 20  # Padding
    words = status_to_display.split(' ')
    lines = []
    current_line_text = ""
    for word in words:
        test_line = current_line_text + word + " "
        if info_font.size(test_line)[0] <= max_width:
            current_line_text = test_line
        else:
            lines.append(current_line_text.strip())
            current_line_text = word + " "
    lines.append(current_line_text.strip())

    line_y = BOARD_HEIGHT + 10
    for i, line_render_text in enumerate(lines):
        if i >= 3 and INFO_PANEL_HEIGHT < 100: break # Crude limit for very short panels
        if line_y + info_font.get_linesize() > SCREEN_HEIGHT: break # Don't draw off panel

        text_surface = info_font.render(line_render_text, True, TEXT_COLOR)
        screen.blit(text_surface, (10, line_y))
        line_y += info_font.get_linesize()

    pygame.display.flip()

# --- Board State Update Logic (Array for physical detection) ---
def update_board_state_for_move(board_state_array: np.ndarray, board_before_move: chess.Board, move: chess.Move) -> np.ndarray | None:
    if board_state_array is None: 
        print("[WARN] update_board_state_for_move: board_state_array is None.")
        return None
    if not PHONE_DETECTION_ENABLED: # Should not be called if disabled, but good check
        return board_state_array 

    print(f"[DEBUG] Updating board state array for move: {move.uci()}")
    new_state = board_state_array.copy()

    # Color of the piece that *just* moved (whose turn it WAS)
    # If it's currently Black's turn on board_before_move, White just moved.
    moving_piece_color_val = 1 if board_before_move.turn == chess.BLACK else -1

    from_r, from_f = map_chess_square_to_array_indices(chess.square_name(move.from_square))
    to_r, to_f = map_chess_square_to_array_indices(chess.square_name(move.to_square))

    if from_r is None or to_r is None:
        print("[WARN] update_board_state_for_move: Could not map move squares.")
        return board_state_array 

    new_state[from_r, from_f] = 0 # Empty origin
    new_state[to_r, to_f] = moving_piece_color_val # Place piece on destination

    # Handle en passant capture (remove pawn from actual captured square)
    if board_before_move.is_en_passant(move):
        print("[DEBUG] update_board_state_for_move: Handling en passant.")
        # The captured pawn is on the 'to' file, but 'from' rank of the moving pawn
        cap_f_idx = chess.square_file(move.to_square)
        cap_r_idx_chess = chess.square_rank(move.from_square) # Chess rank (0-7)
        
        # Convert to array indices for captured pawn
        # If white captures en passant (e.g., e5xd6), white moves from rank 4 (idx 3) to rank 5 (idx 2).
        # Black pawn was on rank 4 (idx 3) at file D.
        # So captured pawn's array row is 7 - cap_r_idx_chess.
        cap_r_arr = 7 - cap_r_idx_chess
        cap_f_arr = cap_f_idx

        if 0 <= cap_r_arr <= 7 and 0 <= cap_f_arr <= 7:
             # Check if the square actually had an opponent pawn
            if new_state[cap_r_arr, cap_f_arr] == -moving_piece_color_val: # Opponent color
                 new_state[cap_r_arr, cap_f_arr] = 0
                 print(f"[DEBUG] En passant removed opponent pawn at array index: [{cap_r_arr}][{cap_f_arr}]")
            # else: The board_state_array might have been slightly off, but move is legal.
        else: print("[WARN] Could not determine en passant captured square index for array.")

    # Handle castling rook movement in the array
    if board_before_move.is_castling(move):
        print("[DEBUG] update_board_state_for_move: Handling castling.")
        rook_map = {
            chess.G1: (chess.H1, chess.F1), chess.C1: (chess.A1, chess.D1), # White
            chess.G8: (chess.H8, chess.F8), chess.C8: (chess.A8, chess.D8)  # Black
        }
        if move.to_square in rook_map: # move.to_square is king's destination
            rook_from_sq, rook_to_sq = rook_map[move.to_square]
            rf_r, rf_f = map_chess_square_to_array_indices(chess.square_name(rook_from_sq))
            rt_r, rt_f = map_chess_square_to_array_indices(chess.square_name(rook_to_sq))
            if rf_r is not None and rt_r is not None:
                # Check if rook was indeed at origin in our array representation
                if new_state[rf_r, rf_f] == moving_piece_color_val : # Or board_state_array[rf_r, rf_f]
                    new_state[rf_r, rf_f] = 0       # Empty rook origin
                    new_state[rt_r, rt_f] = moving_piece_color_val # Place rook
                # else: array might be slightly off, but move is legal. Trust the game logic.
            else: print("[WARN] Could not map castling rook squares for array.")
    return new_state


# --- Game Loop ---
def game_loop_physical(screen, board: chess.Board, engine: chess.engine.SimpleEngine, piece_font, info_font, clock, initial_board_state_array_for_game: np.ndarray):
    running = True
    current_board_state_array = initial_board_state_array_for_game
    current_action_prompt = "" 
    human_move_made_this_turn = False # Used to control display updates at end of main loop

    if current_board_state_array is None and PHONE_DETECTION_ENABLED:
        update_display(screen, board, piece_font, info_font, "Error: Missing initial board state. Exiting.")
        pygame.time.wait(3000)
        return "BOARD_STATE_INIT_FAILED_IN_LOOP"

    print(f"[INFO] Game loop started. Initial Board FEN: {board.fen()}")
    print("-" * 30)

    while running and not board.is_game_over(claim_draw=True):
        human_move_made_this_turn = False 

        # --- Human Player's Turn (e.g., White) ---
        if board.turn == HUMAN_PLAYER_COLOR and running and not board.is_game_over(claim_draw=True):
            if not PHONE_DETECTION_ENABLED:
                update_display(screen, board, piece_font, info_font, "Error: Phone detection disabled.")
                pygame.time.wait(3000); running = False; continue

            # --- State machine for human turn ---
            human_turn_state = "AWAIT_PHOTO_CONFIRM" 
            detection_attempts_current_move = 0
            user_typed_move_str = ""
            last_detected_move_uci = None # To show user what was detected
            _cached_new_board_state_for_detected_move = None # To use if user confirms 'Y'

            valid_move_found_for_turn = False
            while not valid_move_found_for_turn and running:
                
                # Determine and display the current prompt based on state
                if human_turn_state == "AWAIT_PHOTO_CONFIRM":
                    current_action_prompt = "Your turn. Make move, take pic, then press Enter."
                elif human_turn_state == "AWAIT_YN_RESPONSE":
                    current_action_prompt = f"Detected: {last_detected_move_uci}. Correct? (Y/N)"
                elif human_turn_state == "AWAIT_TYPED_MOVE":
                    # Add a blinking cursor effect for typed input
                    cursor_char = "_" if (pygame.time.get_ticks() // 500) % 2 == 0 else " "
                    current_action_prompt = f"Type move (e.g. e2e4): {user_typed_move_str}{cursor_char}"
                
                update_display(screen, board, piece_font, info_font, current_action_prompt)

                action_taken_this_loop_iteration = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False; valid_move_found_for_turn = True; break # Exit all loops for this turn

                    if event.type == pygame.KEYDOWN:
                        action_taken_this_loop_iteration = True # A key relevant to current state was pressed

                        if human_turn_state == "AWAIT_PHOTO_CONFIRM":
                            if event.key == pygame.K_RETURN:
                                detection_attempts_current_move += 1
                                update_display(screen, board, piece_font, info_font, "Processing your move...")
                                pygame.display.flip() # Ensure "Processing..." is shown
                                
                                _detected_uci, _detected_new_state = detect_human_move(current_board_state_array, board) 
                                
                                if _detected_uci and _detected_new_state is not None:
                                    last_detected_move_uci = _detected_uci
                                    _cached_new_board_state_for_detected_move = _detected_new_state
                                    human_turn_state = "AWAIT_YN_RESPONSE"
                                else: 
                                    print("[WARN] Vision failed to detect a move or an ADB/image error occurred.")
                                    if detection_attempts_current_move >= 2:
                                        human_turn_state = "AWAIT_TYPED_MOVE"; user_typed_move_str = ""
                                    else: # Stays in AWAIT_PHOTO_CONFIRM, prompt will re-appear
                                        pass 
                            # else ignore other keys in this state

                        elif human_turn_state == "AWAIT_YN_RESPONSE":
                            if event.key == pygame.K_y:
                                try:
                                    move_obj = board.parse_uci(last_detected_move_uci)
                                    if move_obj in board.legal_moves:
                                        print(f"[INFO] <<< Human move '{last_detected_move_uci}' CONFIRMED by user & VALIDATED >>>")
                                        current_board_state_array = _cached_new_board_state_for_detected_move 
                                        board.push(move_obj)
                                        valid_move_found_for_turn = True
                                        human_move_made_this_turn = True 
                                    else: # Confirmed move is illegal (e.g. vision hallucination)
                                        print(f"[ERROR] User confirmed move '{last_detected_move_uci}', but it's illegal. Vision/logic error.")
                                        if detection_attempts_current_move >= 2:
                                            human_turn_state = "AWAIT_TYPED_MOVE"; user_typed_move_str = ""
                                        else:
                                            human_turn_state = "AWAIT_PHOTO_CONFIRM" # Try photo again
                                except ValueError: # Should be rare if detect_human_move provides valid UCI
                                     print(f"[CRITICAL ERROR] UCI parse error for detected move '{last_detected_move_uci}'.")
                                     if detection_attempts_current_move >= 2: # Treat as failed attempt
                                        human_turn_state = "AWAIT_TYPED_MOVE"; user_typed_move_str = ""
                                     else:
                                        human_turn_state = "AWAIT_PHOTO_CONFIRM"
                            elif event.key == pygame.K_n:
                                print(f"[INFO] User rejected detected move: '{last_detected_move_uci}'. Attempts: {detection_attempts_current_move}")
                                if detection_attempts_current_move >= 2:
                                    human_turn_state = "AWAIT_TYPED_MOVE"; user_typed_move_str = ""
                                else:
                                    human_turn_state = "AWAIT_PHOTO_CONFIRM" 
                            # else ignore other keys

                        elif human_turn_state == "AWAIT_TYPED_MOVE":
                            if event.key == pygame.K_RETURN:
                                if user_typed_move_str: # Process if not empty
                                    try:
                                        move_obj = board.parse_uci(user_typed_move_str)
                                        if move_obj in board.legal_moves:
                                            print(f"[INFO] <<< Human move TYPED & VALIDATED: '{user_typed_move_str}' >>>")
                                            # board is state *before* this typed move for update_board_state_for_move
                                            current_board_state_array = update_board_state_for_move(current_board_state_array, board, move_obj)
                                            board.push(move_obj)
                                            valid_move_found_for_turn = True
                                            human_move_made_this_turn = True
                                            user_typed_move_str = "" 
                                        else:
                                            print(f"[ERROR] Typed move '{user_typed_move_str}' is ILLEGAL.")
                                            update_display(screen, board, piece_font, info_font, f"Illegal: {user_typed_move_str}. Type move again:")
                                            user_typed_move_str = "" 
                                    except ValueError:
                                        print(f"[ERROR] Typed move '{user_typed_move_str}' is INVALID UCI format.")
                                        update_display(screen, board, piece_font, info_font, f"Invalid format: {user_typed_move_str}. Type move again:")
                                        user_typed_move_str = "" 
                                # else: empty input on Enter, do nothing, let prompt refresh
                            elif event.key == pygame.K_BACKSPACE:
                                user_typed_move_str = user_typed_move_str[:-1]
                            elif len(user_typed_move_str) < 5: # Max typical UCI move length (e.g., e7e8q)
                                # Only allow ASCII alphanumeric characters for typed input
                                if event.unicode.isascii() and event.unicode.isalnum():
                                     user_typed_move_str += event.unicode.lower() # Standardize to lower
                # End of KEYDOWN event processing
                if not running: break # Exit event loop if QUIT occurred
            # End of event_for loop for one iteration

            if not running: break # Exit 'while not valid_move_found_for_turn' if QUIT
            
            # If no relevant key was pressed for the current state, ensure the loop ticks and display updates with cursor
            if not action_taken_this_loop_iteration or human_turn_state == "AWAIT_TYPED_MOVE": 
                 # Always tick for typed move state to update cursor, or if just waiting
                 clock.tick(30) 
            
            # If a move was successfully found and made, update display for engine's turn
            if valid_move_found_for_turn and human_move_made_this_turn and running:
                update_display(screen, board, piece_font, info_font, "Move accepted. Engine's turn...")
                pygame.time.wait(500) # Brief pause to show confirmation

        # --- Engine Player's Turn (e.g., Black) ---
        if board.turn == ENGINE_PLAYER_COLOR and running and not board.is_game_over(claim_draw=True):
            current_action_prompt = "Engine's turn... Thinking..."
            update_display(screen, board, piece_font, info_font, current_action_prompt)

            engine_move = None
            try:
                if engine:
                    result = engine.play(board, chess.engine.Limit(time=1.0)) 
                    if result.move:
                        engine_move = result.move
                        print(f"[INFO] >>> Engine proposes move: {engine_move.uci()} <<<") 

                        if current_board_state_array is not None and PHONE_DETECTION_ENABLED:
                            # Pass board *before* engine_move is pushed for correct context in update_board_state_for_move
                            current_board_state_array = update_board_state_for_move(current_board_state_array, board, engine_move)
                        
                        board.push(engine_move) 
                        print(f"[INFO] Engine move {engine_move.uci()} pushed to logical board.")

                        current_action_prompt = f"Engine played: {engine_move.uci()}. Make this move on physical board, then press Enter."
                        update_display(screen, board, piece_font, info_font, current_action_prompt) 
                        pygame.time.wait(100) # Small delay to help ensure message is visually processed

                        engine_move_physically_confirmed = False
                        while not engine_move_physically_confirmed and running:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    running = False; engine_move_physically_confirmed = True; break
                                if event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_RETURN:
                                        engine_move_physically_confirmed = True
                            if not running: break
                            clock.tick(30) 
                        
                        if not running: break 
                    else: 
                        print("[WARN] Engine returned no move.")
                        current_action_prompt = "Engine error: No move. Game cannot continue."
                        update_display(screen, board, piece_font, info_font, current_action_prompt)
                        running = False 
                else: # Engine object is None
                    print("[CRITICAL] Engine object is None."); running = False
                    update_display(screen, board, piece_font, info_font, "Critical Error: Engine not available.")
                    pygame.time.wait(3000)
            except Exception as e:
                print(f"[CRITICAL] Error during engine move: {e}"); running = False
                error_msg_short = str(e)[:50] # Avoid overly long error messages in Pygame window
                update_display(screen, board, piece_font, info_font, f"Engine Error: {error_msg_short}")
                pygame.time.wait(3000)
            
            if not running: break # Break main game loop if quit or error during engine's turn

        # --- Loop End / Check Game Over ---
        if board.is_game_over(claim_draw=True):
            running = False # End game loop naturally
        
        if running : clock.tick(15) # Main game loop tick only if still running

    # --- Game Over Display ---
    final_status_text = "Game Interrupted"
    game_ended_naturally = False
    if board.is_game_over(claim_draw=True):
        result_text = board.result(claim_draw=True)
        if result_text == "1-0": final_status_text = "Game Over: White wins!"
        elif result_text == "0-1": final_status_text = "Game Over: Black wins!"
        elif result_text == "1/2-1/2": final_status_text = "Game Over: Draw!"
        else: final_status_text = f"Game Over: {result_text}"
        game_ended_naturally = True
    elif not running and not game_ended_naturally: # Game aborted by user (e.g., QUIT event)
        final_status_text = "Game Aborted by User"
    
    print("-" * 30); print(f"[INFO] {final_status_text}"); print("-" * 30)
    
    # Final screen loop (ensure display stays if Pygame is init)
    if pygame.get_init() and screen: # Check if screen is still valid
        update_display(screen, board if board else chess.Board(), piece_font, info_font, final_status_text)
        final_screen_loop_active = True
        while final_screen_loop_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: final_screen_loop_active = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                        final_screen_loop_active = False
            if final_screen_loop_active: clock.tick(15) # Keep responsive

    return "GAME_ENDED" if game_ended_naturally else "GAME_ABORTED"


# --- Setup Pygame and Engine ---
def setup_pygame_and_engine(difficulty):
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({"Skill Level": max(0, min(20, difficulty))}) # Ensure skill level is within 0-20
        print(f"[INFO] Stockfish engine started with skill level {difficulty}.")
    except Exception as e:
        print(f"[CRITICAL] Error starting Stockfish engine: {e}")
        # Use a more robust way to show error if Tk is not mainlooping here
        root_err = tk.Tk(); root_err.withdraw() # Create and hide temporary Tk root
        messagebox.showerror("Engine Error", f"Could not start Stockfish: {e}\nPlease check STOCKFISH_PATH in game_logic.py.", parent=root_err)
        root_err.destroy()
        return None, None, None, None, None # Return tuple of Nones
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Physical Chess Robot Control")
    clock = pygame.time.Clock()
    
    # Font loading (more robust)
    font_path = None 
    try: 
        font_path_candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DejaVuSans.ttf")
        if os.path.exists(font_path_candidate):
            font_path = font_path_candidate
            print(f"[INFO] Using font: {font_path}")
        else:
            print(f"[WARN] Font DejaVuSans.ttf not found at '{font_path_candidate}'. Will use Pygame default.")
    except Exception as e_font_path:
        print(f"[WARN] Error determining font path: {e_font_path}. Will use Pygame default.")

    try:
        piece_font_size = int(SQUARE_SIZE * 0.8)
        piece_font = pygame.font.Font(font_path, piece_font_size) # font_path can be None for default
        
        info_font_size = 24 # Default size for info panel
        if INFO_PANEL_HEIGHT < 60: info_font_size = 18 # Smaller for very short panel
        elif INFO_PANEL_HEIGHT < 80: info_font_size = 20
        info_font = pygame.font.Font(font_path, info_font_size)
        print(f"[INFO] Piece font size: {piece_font_size}, Info font size: {info_font_size}")
    except Exception as e: # Fallback if specific font loading fails
        print(f"[WARN] Error loading font (path: '{font_path}'): {e}. Using default system font for all.")
        piece_font = pygame.font.Font(None, int(SQUARE_SIZE * 0.9)) 
        info_font = pygame.font.Font(None, 30) 

    return engine, screen, piece_font, info_font, clock

# --- Generate 8x8 Integer Array from FEN ---
def generate_state_array_from_fen(fen_string: str) -> np.ndarray | None:
    try:
        board_val = chess.Board(fen_string) 
    except ValueError:
        print(f"[ERROR] generate_state_array_from_fen: Invalid FEN string '{fen_string}'")
        return None
        
    state_array = np.zeros((8, 8), dtype=np.int8)
    for r_chess in range(8): # chess rank 0 (board's 1st rank) to 7 (board's 8th rank)
        for f_chess in range(8): # chess file 0 (a-file) to 7 (h-file)
            square = chess.square(f_chess, r_chess)
            piece = board_val.piece_at(square)
            
            arr_r = 7 - r_chess # Map chess rank to array row (0..7 -> 7..0)
            arr_f = f_chess     # Chess file to array col (0..7 -> 0..7)
            
            if piece:
                state_array[arr_r, arr_f] = 1 if piece.color == chess.WHITE else -1
            else:
                state_array[arr_r, arr_f] = 0
    return state_array

# --- Main Game Start Function (Called by UI) ---
def start_game_from_photo(difficulty, initial_fen, initial_board_state_array_from_photo):
    print(f"\n[INFO] --- Starting Game via Photo Detection (Initial FEN: {initial_fen}) ---")

    if not PHONE_DETECTION_ENABLED:
         # Create a temporary Tk root for the messagebox if not in a Tk mainloop
         root_msg = tk.Tk(); root_msg.withdraw()
         messagebox.showerror("Setup Error", "Physical board detection module failed to load. Check console.", parent=root_msg)
         root_msg.destroy(); return "MODULE_LOAD_FAILED"
    
    adb_check_root = tk.Tk(); adb_check_root.withdraw()
    if not check_adb_device():
        messagebox.showerror("ADB Error", "ADB device check failed. Ensure phone is connected, USB Debugging is ON, and PC is trusted. Check console for details.", parent=adb_check_root)
        adb_check_root.destroy(); return "ADB_CHECK_FAILED"
    adb_check_root.destroy()

    dialog_parent_window = tk.Tk(); dialog_parent_window.withdraw() # For simpledialog FEN correction
    engine, screen, piece_font, info_font, clock = setup_pygame_and_engine(difficulty)
    
    if engine is None or screen is None: # Error already shown by setup_pygame_and_engine
        if pygame.get_init(): pygame.quit() # Ensure Pygame is quit if it was initialized
        if dialog_parent_window.winfo_exists(): dialog_parent_window.destroy()
        return "SETUP_FAILED"

    board = None # The main game board object
    final_fen_for_game = initial_fen
    fen_was_corrected_by_user = False # Track if user manually changed FEN

    print("[INFO] Setting up board from detected FEN and confirming with user in Pygame window...")
    if not final_fen_for_game: # Should be caught by UI, but double check
        if pygame.get_init(): pygame.quit()
        messagebox.showerror("Internal Error", "No FEN received from UI setup.", parent=dialog_parent_window)
        dialog_parent_window.destroy(); return "MISSING_FEN"

    try:
        try: board_to_display = chess.Board(final_fen_for_game) # For initial display
        except ValueError: # Invalid FEN from UI
            if pygame.get_init(): pygame.quit()
            messagebox.showerror("Initial FEN Error", f"Detected FEN is invalid:\n{final_fen_for_game}\nCannot start game.", parent=dialog_parent_window)
            dialog_parent_window.destroy(); return "INVALID_FEN_DETECTED"

        # --- FEN Confirmation in Pygame Window ---
        update_display(screen, board_to_display, piece_font, info_font, f"Confirm: (Y/N)")
        
        fen_confirmation_pending = True
        user_confirmed_fen_correct = False
        user_aborted_fen_confirm = False

        while fen_confirmation_pending:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    fen_confirmation_pending = False; user_aborted_fen_confirm = True; break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        user_confirmed_fen_correct = True; fen_confirmation_pending = False; break
                    elif event.key == pygame.K_n:
                        user_confirmed_fen_correct = False; fen_confirmation_pending = False; break
            if not fen_confirmation_pending: break # Exit if Y, N, or QUIT
            clock.tick(15) # Keep window responsive

        if user_aborted_fen_confirm:
            if engine: engine.quit()
            if pygame.get_init(): pygame.quit()
            dialog_parent_window.destroy(); return "USER_ABORTED_FEN_CONFIRM"

        if not user_confirmed_fen_correct: # User pressed 'N'
            pygame.display.quit() # Temporarily close Pygame window for Tkinter dialog
            corrected_fen_str = simpledialog.askstring("Correct FEN", 
                                                       "Board state was NOT correct.\nPlease paste or type the correct FEN:",
                                                       parent=dialog_parent_window, 
                                                       initialvalue=board_to_display.fen())
            # Reinitialize Pygame screen after Tkinter dialog
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Physical Chess Robot Control")
            # Need to re-setup fonts if they were complex, but for now, assume piece_font and info_font are still valid.

            if corrected_fen_str:
                try:
                    board_temp_check = chess.Board(corrected_fen_str) # Validate new FEN
                    final_fen_for_game = corrected_fen_str
                    fen_was_corrected_by_user = True
                    board_to_display = board_temp_check # Update display board
                    print(f"[INFO] FEN corrected by user to: {final_fen_for_game}")
                    update_display(screen, board_to_display, piece_font, info_font, "FEN Corrected. Starting...")
                except ValueError:
                    update_display(screen, board_to_display, piece_font, info_font, "Invalid corrected FEN. Using previous.")
                    print(f"[WARN] Invalid FEN entered by user ('{corrected_fen_str}'). Using previous: {final_fen_for_game}")
                    # final_fen_for_game remains the one before this failed correction attempt
                    fen_was_corrected_by_user = False # As correction failed
            else: # User cancelled the FEN input dialog
                update_display(screen, board_to_display, piece_font, info_font, "Correction cancelled. Using previous FEN.")
                print("[INFO] FEN correction cancelled by user. Using previous FEN.")
                # final_fen_for_game remains the one displayed before 'N' was pressed
                fen_was_corrected_by_user = False
            pygame.time.wait(1500) # Show message briefly

        print(f"[INFO] Final FEN for game confirmed: {final_fen_for_game}")
        board = chess.Board(final_fen_for_game) # Create the definitive game board
        if not board.is_valid():
            update_display(screen, board_to_display, piece_font, info_font, "Error: Final FEN is illegal!")
            pygame.time.wait(3000)
            if engine: engine.quit(); pygame.quit(); dialog_parent_window.destroy(); return "ILLEGAL_FEN_STATE"
        print(f"[INFO] Confirmed FEN '{board.fen()}' is valid.")

    except Exception as e_fen_setup: 
        print(f"[CRITICAL] Error during FEN setup phase: {e_fen_setup}"); import traceback; traceback.print_exc()
        # Try to show error in Pygame if possible
        if pygame.get_init() and screen:
            try: update_display(screen, chess.Board(), piece_font, info_font, f"FEN Setup Error: {str(e_fen_setup)[:50]}")
            except: pass # If display itself fails
            pygame.time.wait(4000)
        if engine: engine.quit()
        if pygame.get_init(): pygame.quit()
        dialog_parent_window.destroy(); return "FEN_SETUP_EXCEPTION"

    # --- Determine Initial Physical Board State Array ---
    initial_board_state_array_for_game_final = None; print("-" * 30)
    if fen_was_corrected_by_user:
        print(f"[INFO] FEN was corrected. Generating board state array from: {final_fen_for_game}")
        initial_board_state_array_for_game_final = generate_state_array_from_fen(final_fen_for_game)
        if initial_board_state_array_for_game_final is None: 
            update_display(screen, board, piece_font, info_font, "Error: State gen from FEN failed!")
            pygame.time.wait(3000)
            engine.quit(); pygame.quit(); dialog_parent_window.destroy(); return "STATE_FROM_FEN_FAILED"
    else: 
        print("[INFO] Using initial board state array from the photo (FEN confirmed or correction reverted/cancelled).")
        initial_board_state_array_for_game_final = initial_board_state_array_from_photo
        if initial_board_state_array_for_game_final is None:
             update_display(screen, board, piece_font, info_font, "Error: Initial array missing!")
             pygame.time.wait(3000)
             engine.quit(); pygame.quit(); dialog_parent_window.destroy(); return "MISSING_INITIAL_ARRAY"

    # Final validation of the array before starting game loop
    if not isinstance(initial_board_state_array_for_game_final, np.ndarray) or \
       initial_board_state_array_for_game_final.dtype != np.int8 or \
       initial_board_state_array_for_game_final.shape != (8,8):
        update_display(screen, board, piece_font, info_font, "Error: Board array invalid!")
        pygame.time.wait(3000)
        engine.quit(); pygame.quit(); dialog_parent_window.destroy(); return "INVALID_FINAL_ARRAY_FORMAT"
    
    print("[INFO] Initial physical board state array determined successfully."); print("-" * 30)
    
    update_display(screen, board, piece_font, info_font, "Starting game...") 
    pygame.time.wait(1000) # Brief pause

    game_status = "UNKNOWN_ERROR"
    try:
        game_status = game_loop_physical(screen, board, engine, piece_font, info_font, clock, initial_board_state_array_for_game_final)
    except Exception as e:
        print(f"[CRITICAL] Unhandled error during game loop: {e}"); import traceback; traceback.print_exc()
        game_status = "GAME_LOOP_CRASH"
        if pygame.get_init() and screen:
            try: update_display(screen, board if board else chess.Board(), piece_font, info_font, f"Crash: {str(e)[:60]}")
            except: pass 
            pygame.time.wait(5000)
    finally:
        if engine and engine.is_running(): engine.quit()
        if pygame.get_init(): pygame.quit()
        if dialog_parent_window.winfo_exists(): dialog_parent_window.destroy()
        print(f"[INFO] Game finished with status: {game_status}"); print("-" * 30)
        return game_status