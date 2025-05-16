import cv2
import cv2.aruco as aruco
import numpy as np
import os

def parse_fen_to_board(fen_piece_placement):
    """
    Parses the piece placement part of a FEN string (with '-' as rank separator)
    into an 8x8 list of lists representing the board.
    ' ' is used for empty squares.
    """
    board_matrix = []
    ranks = fen_piece_placement.split('-')
    if len(ranks) != 8:
        # Skip this image if FEN is invalid, or handle error as preferred
        print(f"Warning: FEN string part '{fen_piece_placement}' must contain 8 ranks. Skipping.")
        raise ValueError(f"FEN string part '{fen_piece_placement}' must contain 8 ranks separated by '-'. Found {len(ranks)}.")

    for rank_fen in ranks:
        board_row = []
        for char in rank_fen:
            if char.isdigit():
                board_row.extend([' '] * int(char))
            else:
                board_row.append(char)
        if len(board_row) != 8:
            # Skip this image if FEN rank is invalid
            print(f"Warning: FEN rank '{rank_fen}' does not expand to 8 squares. Skipping.")
            raise ValueError(f"FEN rank '{rank_fen}' does not expand to 8 squares. Current row: {board_row}")
        board_matrix.append(board_row)
    return board_matrix

def get_piece_folder_name(piece_char):
    """
    Determines the folder name for a given piece character.
    Uses prefixes 'w' for white and 'b' for black to ensure distinct folder names
    on case-insensitive file systems.
    Standard FEN piece characters are used for logic (e.g., 'P', 'p').
    Folder names will be like 'wP', 'bP', 'wK', 'bK', 'empty'.
    """
    if piece_char == ' ':
        return 'empty'
    elif piece_char.islower():  # Black piece (e.g., p, r, n, b, q, k)
        return 'b' + piece_char.upper() # Becomes 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK'
    else:  # White piece (e.g., P, R, N, B, Q, K)
        return 'w' + piece_char # Becomes 'wP', 'wR', 'wN', 'wB', 'wQ', 'wK'

def process_chessboard_image(image_path, output_base_dir):
    """
    Processes a chessboard image:
    1. Extracts FEN from filename.
    2. Detects ArUco markers to find chessboard corners.
    3. Warps and crops the chessboard.
    4. Crops each of the 64 cells.
    5. Saves cells into specific folders (e.g., wP, bP, empty)
       with numerical filenames (1.png, 2.png, ...).
    """
    print(f"\n--- Processing image: {image_path} ---")
    filename = os.path.basename(image_path)
    fen_from_filename = os.path.splitext(filename)[0]
    
    # Check for invalid characters in FEN that might make invalid folder names
    # For example, FEN strings use '/', but filenames might use '-' instead.
    # The fen_specific_output_dir uses fen_from_filename, which might have these characters.
    # Let's ensure fen_from_filename is safe for directory naming.
    # Common problematic characters for folder names: / \ : * ? " < > |
    # FEN from filename typically uses '-' as rank separator which is fine for folder names.
    # If it used '/' it would be an issue. The script seems to handle '-' in filenames.

    print(f"Using FEN from filename: {fen_from_filename.replace('-', '/')}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}. Skipping.")
        return

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    try:
        aruco_parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray_img)
    except AttributeError:
        print("Using legacy ArUco detection method.")
        parameters = aruco.DetectorParameters_create()
        marker_corners, marker_ids, _ = aruco.detectMarkers(gray_img, aruco_dictionary, parameters=parameters)

    if marker_ids is None or len(marker_ids) < 4:
        detected_count = len(marker_ids) if marker_ids is not None else 0
        print(f"Error: Less than 4 ArUco markers detected in {filename}. Found {detected_count}. Skipping.")
        return

    detected_markers_map = {marker_id[0]: corner[0] for marker_id, corner in zip(marker_ids, marker_corners)}
    required_ids = [0, 1, 2, 3]
    for req_id in required_ids:
        if req_id not in detected_markers_map:
            print(f"Error: Required ArUco marker ID {req_id} not found in {filename}. Skipping.")
            return

    try:
        board_top_left = detected_markers_map[0][2]
        board_top_right = detected_markers_map[1][3]
        board_bottom_right = detected_markers_map[2][0]
        board_bottom_left = detected_markers_map[3][1]
    except IndexError:
        print(f"Error: Could not extract necessary marker corners in {filename}. Skipping.")
        return
    except KeyError as e:
        print(f"Error: Critical marker ID {e} missing in {filename}. Skipping.")
        return

    source_points = np.array([board_top_left, board_top_right, board_bottom_right, board_bottom_left], dtype="float32")
    warped_board_size = 4000 # As per your preference
    destination_points = np.array([
        [0, 0], [warped_board_size - 1, 0],
        [warped_board_size - 1, warped_board_size - 1], [0, warped_board_size - 1]
    ], dtype="float32")

    perspective_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    warped_chessboard_img = cv2.warpPerspective(img, perspective_matrix, (warped_board_size, warped_board_size))
    
    try:
        board_layout = parse_fen_to_board(fen_from_filename)
    except ValueError as e:
        print(f"Error parsing FEN for {filename}: {e}. Skipping.")
        return

    fen_specific_output_dir = os.path.join(output_base_dir, fen_from_filename)
    os.makedirs(fen_specific_output_dir, exist_ok=True)

    base_white_fen_chars = ['P', 'R', 'N', 'B', 'Q', 'K']
    base_black_fen_chars = ['p', 'r', 'n', 'b', 'q', 'k']

    actual_folder_names_to_create = ['empty']
    for p_char in base_white_fen_chars:
        actual_folder_names_to_create.append(get_piece_folder_name(p_char))
    for p_char in base_black_fen_chars:
        actual_folder_names_to_create.append(get_piece_folder_name(p_char))
    
    actual_folder_names_to_create = sorted(list(set(actual_folder_names_to_create)))

    for folder_name in actual_folder_names_to_create:
        os.makedirs(os.path.join(fen_specific_output_dir, folder_name), exist_ok=True)

    piece_image_counters = {folder_name: 0 for folder_name in actual_folder_names_to_create}
    cell_size = warped_board_size // 8

    for row_idx in range(8):
        for col_idx in range(8):
            start_y, end_y = row_idx * cell_size, (row_idx + 1) * cell_size
            start_x, end_x = col_idx * cell_size, (col_idx + 1) * cell_size
            cell_img = warped_chessboard_img[start_y:end_y, start_x:end_x]

            piece_on_square = board_layout[row_idx][col_idx]
            target_subfolder = get_piece_folder_name(piece_on_square)

            piece_image_counters[target_subfolder] += 1
            current_image_count = piece_image_counters[target_subfolder]
            
            cell_image_filename = f"{current_image_count}.png"
            path_to_save_cell = os.path.join(fen_specific_output_dir, target_subfolder, cell_image_filename)
            
            try:
                cv2.imwrite(path_to_save_cell, cell_img)
            except Exception as e:
                print(f"Error saving cell image {path_to_save_cell} for {filename}: {e}")

    print(f"Successfully processed and saved cells for {filename} in {fen_specific_output_dir}")
    print(f"Image counts per folder for {filename}: {piece_image_counters}")

# --- Main execution ---
if __name__ == "__main__":
    # PLEASE REPLACE THE FOLLOWING PATHS WITH YOUR ACTUAL PATHS
    # This should now be the folder containing your full chessboard images
    input_images_folder = r""# Replace with your actual path
    # This is the base directory where FEN-named subfolders will be created
    base_output_directory = r""# Replace with your actual path

    # Define allowed image extensions
    allowed_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    if not os.path.isdir(input_images_folder):
         print(f"Error: Input folder path does not exist or is not a directory: {input_images_folder}")
    elif not os.path.exists(base_output_directory):
        try:
            os.makedirs(base_output_directory, exist_ok=True)
            print(f"Created base output directory: {base_output_directory}")
        except OSError as e:
            print(f"Error: Could not create base output directory {base_output_directory}: {e}")
            exit()
    elif not os.path.isdir(base_output_directory):
        print(f"Error: Base output path {base_output_directory} exists but is not a directory.")
    else:
        print(f"Processing images from folder: {input_images_folder}")
        print(f"Output will be saved in subfolders under: {base_output_directory}")
        
        processed_count = 0
        skipped_count = 0
        
        for filename in os.listdir(input_images_folder):
            if filename.lower().endswith(allowed_extensions):
                image_file_path = os.path.join(input_images_folder, filename)
                try:
                    process_chessboard_image(image_file_path, base_output_directory)
                    processed_count += 1
                except Exception as e:
                    print(f"An unexpected error occurred while processing {filename}: {e}. Skipping.")
                    skipped_count +=1
            else:
                # Optional: print a message for non-image files or simply ignore them
                # print(f"Skipping non-image file: {filename}")
                pass
        
        print(f"\n--- Batch processing complete ---")
        print(f"Successfully processed images: {processed_count}")
        print(f"Skipped images (due to errors or non-image format): {skipped_count + (len(os.listdir(input_images_folder)) - processed_count - skipped_count)}") # Also count non-image files as skipped implicitly