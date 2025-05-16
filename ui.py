# ui.py
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import os
import tempfile
import shutil
import numpy as np # For type hinting if needed, and consistency

# Import game logic
from game_logic import start_game_from_photo, PHONE_DETECTION_ENABLED

# Import vision/detection functions needed for setup
try:
    from image_to_fen import get_fen_from_image_path, get_board_state_array_from_image_path, APP_TEMP_PARENT_DIR # Added get_board_state_array_from_image_path
    from physical_board_detector import ( check_adb_device, get_newest_photo_from_phone, pull_photo_from_phone )
    VISION_LOADED = True
except ImportError as e:
    print(f"[UI ERROR] Failed to import vision/detection modules: {e}")
    VISION_LOADED = False
    APP_TEMP_PARENT_DIR = None
    # Add dummy functions if needed for basic UI operation without full functionality
    def get_fen_from_image_path(path): return None
    def get_board_state_array_from_image_path(path): return None
    def check_adb_device(): return False
    def get_newest_photo_from_phone(): return None
    def pull_photo_from_phone(p1, p2): return None


def main_menu():
    root = tk.Tk()
    root.title("Chess Robot - Setup")
    root.geometry("500x280")

    tk.Label(root, text="Chess Robot Setup", font=("Helvetica", 24)).pack(pady=20)
    form_frame = ttk.Frame(root, padding="10"); form_frame.pack(expand=True)

    # Difficulty Selection
    ttk.Label(form_frame, text="Engine Difficulty (0-20):").grid(row=0, column=0, padx=5, pady=10, sticky="w")
    difficulty_setting = tk.IntVar(value=5)
    ttk.Spinbox(form_frame, from_=0, to=20, textvariable=difficulty_setting, width=5, state="readonly", wrap=True).grid(row=0, column=1, padx=5, pady=10, sticky="w")

    # Status Label
    status_label = ttk.Label(form_frame, text="", foreground="red", justify=tk.LEFT)
    status_label.grid(row=1, column=0, columnspan=2, pady=10, sticky="w")

    # Check required modules
    error_messages = []
    if not PHONE_DETECTION_ENABLED: error_messages.append("Physical detection module failed to load (game_logic).")
    if not VISION_LOADED: error_messages.append("Vision/FEN module failed to load (image_to_fen/physical_board_detector imports in UI).")
    if error_messages: status_label.config(text="ERROR:\n" + "\n".join(error_messages) + "\nCannot start game."); can_start = False
    else: status_label.config(text="Required modules loaded.", foreground="green"); can_start = True

    # Buttons
    button_frame = ttk.Frame(root, padding="10"); button_frame.pack(pady=10)
    start_button = ttk.Button(button_frame, text="Start Game", command=lambda: start_game(difficulty_setting.get(), root))
    start_button.pack(side=tk.LEFT, padx=10)
    ttk.Button(button_frame, text="Help", command=lambda: show_help(root)).pack(side=tk.LEFT, padx=10)
    if not can_start: start_button.config(state=tk.DISABLED)

    def show_help(parent_window):
        help_text = (
            "Starting a Game:\n\n"
            "1. Set up the desired starting position on your physical chessboard.\n"
            "2. Ensure the latest photo in your phone's camera roll is a clear picture of this board setup.\n"
            "3. Adjust the Engine Difficulty (0-20).\n"
            "4. Click 'Start Game'.\n\n"
            "Confirmation:\n"
            "The app will analyze the photo to detect the board state (FEN).\n"
            "Confirm if the detected state shown in Pygame is correct (correct FEN if needed).\n"
            "The system will use this initial photo for the board state unless you manually correct the FEN.\n\n"
            "Gameplay:\n"
            "Follow console prompts:\n"
            " - Take photo & press Enter after YOUR move.\n"
            " - Make ENGINE's move physically when prompted.\n\n"
            "Setup Requirements:\n"
            "- ADB, Phone (USB Debugging ON, PC Trusted).\n"
            "- Python libraries installed.\n"
            "- Correct Stockfish path (game_logic.py).\n"
            "- Correct ML model path (image_to_fen.py)."
        )
        messagebox.showinfo("Help & Setup Information", help_text, parent=parent_window)

    def start_game(selected_difficulty, current_setup_root):
        print(f"[UI] Starting Game. Difficulty: {selected_difficulty}")
        if not PHONE_DETECTION_ENABLED or not VISION_LOADED: messagebox.showerror("Error", "Required modules not loaded. Check console for details.", parent=current_setup_root); return
        if not check_adb_device(): messagebox.showerror("ADB Error", "ADB device check failed. Check console.", parent=current_setup_root); return
        if not APP_TEMP_PARENT_DIR: messagebox.showerror("Config Error", "Vision temporary directory not set up (image_to_fen.APP_TEMP_PARENT_DIR).", parent=current_setup_root); return

        initial_fen = None
        initial_board_state_array = None # Variable to hold the array
        data_obtained = False # Combined flag for FEN and array
        temp_dir_unique_session = None

        try:
            while not data_obtained:
                local_temp_image_path = None
                if temp_dir_unique_session and os.path.exists(temp_dir_unique_session):
                    try: shutil.rmtree(temp_dir_unique_session)
                    except OSError as e_rm: print(f"[WARN] Error deleting previous temp dir {temp_dir_unique_session}: {e_rm}")
                    temp_dir_unique_session = None

                proceed = messagebox.askokcancel("Setup Board & Capture Photo",
                                             "1. Ensure physical board is set up for the start of the game.\n"
                                             "2. Ensure LATEST phone photo is of this setup.\n\n"
                                             "Click OK to fetch this image and detect the board state.", parent=current_setup_root)
                if not proceed: print("[UI] User cancelled game start."); return
                current_setup_root.config(cursor="watch"); current_setup_root.update_idletasks()
                print("[UI] Fetching newest photo for initial board state detection...")
                
                image_path_on_phone = get_newest_photo_from_phone()
                if not image_path_on_phone: # Handle case where no photo is found
                    messagebox.showerror("Image Error", "No new photo found on the phone. Please take a picture of the board.", parent=current_setup_root)
                    current_setup_root.config(cursor=""); current_setup_root.update_idletasks()
                    # Allow retry by continuing the loop (data_obtained is false)
                    continue 


                temp_dir_unique_session = tempfile.mkdtemp(prefix="start_game_", dir=APP_TEMP_PARENT_DIR)
                local_temp_image_path = pull_photo_from_phone(image_path_on_phone, temp_dir_unique_session)

                if local_temp_image_path and os.path.exists(local_temp_image_path):
                    print(f"[UI] Processing image {os.path.basename(local_temp_image_path)} for FEN and state array...")
                    initial_fen = get_fen_from_image_path(local_temp_image_path)
                    initial_board_state_array = get_board_state_array_from_image_path(local_temp_image_path)

                    if initial_fen and initial_board_state_array is not None:
                        print(f"[UI] Generated FEN: {initial_fen}")
                        print(f"[UI] Generated initial board state array from the same image.")
                        # Basic validation of the array (optional, but good)
                        if isinstance(initial_board_state_array, np.ndarray) and initial_board_state_array.shape == (8,8):
                            data_obtained = True # Exit loop on success
                        else:
                            print("[UI ERROR] Board state array from image is not a valid 8x8 numpy array.")
                            retry = messagebox.askyesno("State Array Error", "Could generate FEN, but the board state array from image is invalid.\nTry again with a new photo?", parent=current_setup_root)
                            if not retry: print("[UI] User aborted state array retry."); return
                    elif initial_fen and initial_board_state_array is None:
                         print("[UI] FEN generated, but failed to generate board state array from the same image.")
                         retry = messagebox.askyesno("State Array Error", "Could generate FEN but not the board state array from image.\nTry again with a new photo?", parent=current_setup_root)
                         if not retry: print("[UI] User aborted state array retry."); return
                    else: # initial_fen is None (implies initial_board_state_array might also be None or not processed)
                         print("[UI] FEN generation failed from image.")
                         retry = messagebox.askyesno("FEN Error", "Could not determine board state (FEN) from image.\nTry again with a new photo?", parent=current_setup_root)
                         if not retry: print("[UI] User aborted FEN retry."); return
                else:
                    print("[UI] Failed to retrieve/save image from phone.")
                    retry = messagebox.askyesno("Image Error", "Failed to retrieve image from phone.\nTry again?", parent=current_setup_root)
                    if not retry: print("[UI] User aborted image fetch retry."); return
        
        except (RuntimeError, FileNotFoundError, TimeoutError, ConnectionAbortedError, PermissionError) as e:
            messagebox.showerror("Operation Error", f"Error during board setup via photo:\n{e}", parent=current_setup_root)
            return 
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"Unexpected error during board setup: {e}\nCheck console.", parent=current_setup_root)
            import traceback; traceback.print_exc() # For debugging
            return
        finally:
            current_setup_root.config(cursor="")
            if temp_dir_unique_session and os.path.exists(temp_dir_unique_session):
                try:
                    shutil.rmtree(temp_dir_unique_session)
                    print(f"[DEBUG] Cleaned up UI temp dir: {temp_dir_unique_session}")
                except OSError as e_rm:
                    print(f"[WARN] Error deleting UI temp dir {temp_dir_unique_session} in finally: {e_rm}")

        if not data_obtained:
            print("[UI] Board setup via photo aborted or failed after retries."); return

        print("[UI] Initial board state (FEN and array) detected. Proceeding to confirmation and game start...")
        current_setup_root.withdraw()
        # Pass both initial_fen and initial_board_state_array to game_logic
        game_result = start_game_from_photo(
            difficulty=selected_difficulty,
            initial_fen=initial_fen,
            initial_board_state_array_from_photo=initial_board_state_array
        )
        print(f"[UI] Game function returned: {game_result}")
        if current_setup_root.winfo_exists():
            current_setup_root.destroy()

    root.mainloop()