# image_to_fen.py
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import torch
import timm
from torchvision import transforms
from PIL import Image
import chess # For chess square constants and parsing

# --- Configuration ---
# IMPORTANT: Update this path to your actual trained model file
MODEL_PATH = r'' # <<<--- UPDATE PATH
IMAGE_SIZE = 500 # Must match the input size expected by the model
NUM_CLASSES = 13 # bB, bK, bN, bP, bQ, bR, empty, wB, wK, wN, wP, wQ, wR
WARPED_BOARD_SIZE = 4000 # High resolution for cell extraction after warp
CELL_BORDER_MARGIN = 0.1 # Percentage of cell size to ignore around borders

# --- Classes (Must match training order) ---
ALL_CLASSES = sorted(['bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'empty', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR'])
idx_to_class = {i: cls_name for i, cls_name in enumerate(ALL_CLASSES)}

# --- Device Configuration ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Vision Module: Using device: {device}")

# --- Data Transformation for Prediction ---
data_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Application Temporary Directory ---
SCRIPT_DIR_VISION = os.path.dirname(os.path.abspath(__file__))
APP_TEMP_PARENT_DIR = os.path.join(SCRIPT_DIR_VISION, "vision_temp_files")
try:
    os.makedirs(APP_TEMP_PARENT_DIR, exist_ok=True)
    print(f"[INFO] Vision temporary directory ready: {APP_TEMP_PARENT_DIR}")
except OSError as e:
    print(f"[CRITICAL] Could not create vision temporary directory at {APP_TEMP_PARENT_DIR}: {e}")
    APP_TEMP_PARENT_DIR = None

# --- Model Loading ---
def load_model(model_path, num_classes, device_to_use):
    """Loads the trained classification model."""
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    try:
        print(f"[INFO] Attempting to load model weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device_to_use)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        if state_dict is None: raise ValueError("Could not find a valid state_dict in the checkpoint file.")
        model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Successfully loaded model weights onto {device_to_use}.")
    except FileNotFoundError: print(f"[CRITICAL ERROR] Model file not found: {model_path}"); return None
    except Exception as e: print(f"[CRITICAL ERROR] Failed to load model from {model_path}: {e}"); return None
    model = model.to(device_to_use)
    model.eval()
    print("[INFO] Model ready for evaluation.")
    return model

# --- Load Model Globally ---
trained_model = None
if not os.path.exists(MODEL_PATH): print(f"[CRITICAL ERROR] Model path invalid: {MODEL_PATH}. Vision functions will fail.")
else: trained_model = load_model(MODEL_PATH, NUM_CLASSES, device)
if trained_model is None: print(f"[CRITICAL ERROR] Model loading failed. Vision functions unavailable.")

# --- Core Vision Utilities ---
def get_aruco_corners(image):
    """Detects ArUco markers and returns the ordered outer corners of the chessboard."""
    if image is None: print("[ERROR] Cannot detect ArUco corners in None image."); return None
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict_type = aruco.DICT_6X6_250
    try:
        aruco_dictionary = aruco.getPredefinedDictionary(aruco_dict_type)
        aruco_parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray_img)
    except AttributeError:
        print("[DEBUG] Using legacy ArUco detection method (older OpenCV?).")
        aruco_dictionary_obj = cv2.aruco.Dictionary_get(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters_create()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray_img, aruco_dictionary_obj, parameters=parameters)

    if marker_ids is None or len(marker_ids) < 4:
        detected_count = len(marker_ids) if marker_ids is not None else 0
        print(f"[ERROR] ArUco Detection: Found only {detected_count} markers (need 4).")
        return None

    # IMPORTANT: Adjust required_ids_map based on YOUR physical marker setup (IDs and corner indices)
    required_ids_map = { 0: 0, 1: 1, 2: 2, 3: 3 } # Example: {marker_id : corner_index} for TL, TR, BR, BL board corners
    detected_markers_map = {marker_id[0]: corner_set[0] for marker_id, corner_set in zip(marker_ids, marker_corners)}
    board_outer_corners_map = {}

    for req_id, corner_idx in required_ids_map.items():
        if req_id not in detected_markers_map: print(f"[ERROR] Required ArUco marker ID {req_id} not found."); return None
        try: board_outer_corners_map[req_id] = detected_markers_map[req_id][corner_idx]
        except IndexError: print(f"[ERROR] Could not extract corner index {corner_idx} for marker ID {req_id}."); return None

    if len(board_outer_corners_map) != 4: print(f"[ERROR] Failed to map all 4 required marker corners."); return None
    # Order corners: TL, TR, BR, BL (assuming IDs 0, 1, 2, 3 correspond to this order)
    ordered_corners = [ board_outer_corners_map[0], board_outer_corners_map[1], board_outer_corners_map[2], board_outer_corners_map[3] ]
    return np.array(ordered_corners, dtype="float32")

def warp_and_crop_board(image, corners, warped_size):
    """Warps the perspective of the image based on 4 ordered corners (TL, TR, BR, BL)."""
    if image is None or corners is None or len(corners) != 4: print("[ERROR] Cannot warp perspective: Invalid image or corners."); return None
    destination_points = np.array([[0, 0], [warped_size - 1, 0], [warped_size - 1, warped_size - 1], [0, warped_size - 1]], dtype="float32")
    try:
        perspective_matrix = cv2.getPerspectiveTransform(corners, destination_points)
        warped_chessboard_img = cv2.warpPerspective(image, perspective_matrix, (warped_size, warped_size))
        print("[INFO] Perspective warp applied successfully.")
        return warped_chessboard_img
    except Exception as e: print(f"[ERROR] Failed during perspective warp: {e}"); return None

# --- Prediction and State Generation ---
def predict_cell_content(cell_image_pil):
    """Predicts piece/empty class for a single PIL cell image."""
    global trained_model, data_transform, device, idx_to_class
    if trained_model is None: print("[ERROR] Model not available for prediction."); return "empty"
    try: transformed_image = data_transform(cell_image_pil).unsqueeze(0).to(device)
    except Exception as e: print(f"[ERROR] Failed to transform cell image: {e}"); return "empty"
    with torch.no_grad():
        try:
            outputs = trained_model(transformed_image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            predicted_class_idx = preds.item()
            # predicted_confidence = confidence.item() # Optional: use confidence
        except Exception as e: print(f"[ERROR] Model inference failed: {e}"); return "empty"
    predicted_class_name = idx_to_class.get(predicted_class_idx, "empty")
    # print(f"[DEBUG] Cell Prediction: {predicted_class_name} (Conf: {predicted_confidence:.4f})") # Optional
    return predicted_class_name

def board_array_to_fen_placement(board_array_8x8_piece_codes):
    """Converts an 8x8 array of piece codes ('wP', 'bK', 'empty') to FEN piece placement."""
    fen = ""
    for r in range(8):
        empty_count = 0
        for c in range(8):
            piece_code = board_array_8x8_piece_codes[r][c]
            if piece_code == 'empty': empty_count += 1
            else:
                if empty_count > 0: fen += str(empty_count); empty_count = 0
                color = piece_code[0]; piece_type = piece_code[1]
                fen += piece_type.upper() if color == 'w' else piece_type.lower()
        if empty_count > 0: fen += str(empty_count)
        if r < 7: fen += "/"
    return fen

def process_image_to_board_arrays(cv2_image):
    """
    Processes OpenCV image, returns two 8x8 numpy arrays:
    1. Integer state array (0=empty, +1=white, -1=black)
    2. Piece code array ('wP', 'bK', 'empty', etc.)
    Returns (None, None) on failure.
    """
    global trained_model, WARPED_BOARD_SIZE, CELL_BORDER_MARGIN

    if trained_model is None: print("[ERROR] Cannot process image: Model not loaded."); return None, None
    if cv2_image is None: print("[ERROR] Cannot process image: Input image is None."); return None, None

    board_corners = get_aruco_corners(cv2_image)
    if board_corners is None: print("[ERROR] Failed to find board corners."); return None, None
    warped_board = warp_and_crop_board(cv2_image, board_corners, WARPED_BOARD_SIZE)
    if warped_board is None: print("[ERROR] Failed to warp board image."); return None, None

    cell_size_on_warped = WARPED_BOARD_SIZE // 8
    # Use numpy arrays initialized with zeros
    int_state_array = np.zeros((8, 8), dtype=np.int8) # 0=empty, +1=white, -1=black
    piece_code_array = np.full((8, 8), 'empty', dtype=object) # String: 'wP', 'bK', 'empty'

    print("[INFO] Processing board cells for piece identification...")
    cells_processed = 0
    for r_idx in range(8):
        for f_idx in range(8):
            start_y = r_idx * cell_size_on_warped; end_y = (r_idx + 1) * cell_size_on_warped
            start_x = f_idx * cell_size_on_warped; end_x = (f_idx + 1) * cell_size_on_warped
            margin_px_y = int(cell_size_on_warped * CELL_BORDER_MARGIN / 2)
            margin_px_x = int(cell_size_on_warped * CELL_BORDER_MARGIN / 2)
            cell_bgr = warped_board[start_y + margin_px_y : end_y - margin_px_y, start_x + margin_px_x : end_x - margin_px_x]

            if cell_bgr is None or cell_bgr.size == 0:
                print(f"[WARN] Could not extract cell image at [{r_idx}][{f_idx}]")
                # Keep values as 0 and 'empty'
                continue

            cell_rgb = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGB)
            cell_pil = Image.fromarray(cell_rgb)
            predicted_piece_code = predict_cell_content(cell_pil)

            # Populate both arrays
            piece_code_array[r_idx, f_idx] = predicted_piece_code
            if predicted_piece_code == 'empty': int_state_array[r_idx, f_idx] = 0
            elif predicted_piece_code.startswith('w'): int_state_array[r_idx, f_idx] = 1
            elif predicted_piece_code.startswith('b'): int_state_array[r_idx, f_idx] = -1
            cells_processed += 1

    print(f"[INFO] Processed {cells_processed}/64 cells.")
    if cells_processed != 64: print("[WARN] Did not process all 64 cells!")

    return int_state_array, piece_code_array # Return both arrays

# --- High-Level Interface Functions ---
def get_fen_from_image_path(image_path):
    """Takes image path, returns full FEN string (defaulting other fields)."""
    if not os.path.exists(image_path): print(f"[ERROR] Image file not found: {image_path}"); return None
    cv2_img = cv2.imread(image_path)
    if cv2_img is None: print(f"[ERROR] Failed to read image using OpenCV: {image_path}"); return None

    print(f"\n[INFO] Generating FEN from image: {os.path.basename(image_path)}")
    # Get the piece codes needed for FEN generation
    _, piece_codes = process_image_to_board_arrays(cv2_img)

    if piece_codes is None:
        print("[ERROR] FEN generation failed: Could not process image to piece codes.")
        return None

    piece_placement_fen = board_array_to_fen_placement(piece_codes)
    active_color = "w"; castling_availability = "-"; en_passant_target = "-"
    halfmove_clock = "0"; fullmove_number = "1"
    full_fen = f"{piece_placement_fen} {active_color} {castling_availability} {en_passant_target} {halfmove_clock} {fullmove_number}"

    print(f"[INFO] Predicted FEN: {full_fen}")
    try: chess.Board(full_fen); print("[INFO] Generated FEN appears valid.")
    except ValueError as e: print(f"[WARN] Generated FEN might be invalid: {e}\n       FEN: {full_fen}")
    return full_fen

def get_board_state_array_from_image_path(image_path):
     """Takes image path, returns 8x8 integer state array (0, +1, -1)."""
     if not os.path.exists(image_path): print(f"[ERROR] Image file not found: {image_path}"); return None
     cv2_img = cv2.imread(image_path)
     if cv2_img is None: print(f"[ERROR] Failed to read image using OpenCV: {image_path}"); return None

     print(f"\n[INFO] Generating Board State Array from image: {os.path.basename(image_path)}")
     int_state_array, _ = process_image_to_board_arrays(cv2_img) # Ignore piece codes here

     if int_state_array is None: print("[ERROR] Board state array generation failed."); return None
     # print("[DEBUG] Integer State Array Generated:\n", int_state_array) # Optional debug
     return int_state_array

# --- Mapping Utilities ---
def map_array_indices_to_chess_square(r_idx, f_idx):
    """Maps 0-indexed array [row][col] to chess square notation (e.g., [0][0] -> 'a8')."""
    if not (0 <= r_idx <= 7 and 0 <= f_idx <= 7): print(f"[WARN] Invalid array indices for mapping: r={r_idx}, f={f_idx}"); return None
    file_char = chr(ord('a') + f_idx); rank_char = str(8 - r_idx)
    return file_char + rank_char

def map_chess_square_to_array_indices(square_name):
    """Maps chess square notation (e.g., 'e4') to 0-indexed array [row][col]."""
    if not isinstance(square_name, str) or len(square_name) != 2: print(f"[WARN] Invalid square name format: '{square_name}'"); return None, None
    try:
        square_name = square_name.lower(); square_index = chess.parse_square(square_name)
        f_idx = chess.square_file(square_index); r_chess = chess.square_rank(square_index)
        r_idx = 7 - r_chess # Map chess rank 0-7 to array row 7-0
        return r_idx, f_idx
    except ValueError: print(f"[WARN] Could not parse square name: '{square_name}'"); return None, None