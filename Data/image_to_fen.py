import cv2
import cv2.aruco as aruco
import numpy as np
import os
import torch
import timm
from torchvision import transforms
from PIL import Image

# --- Configuration ---
MODEL_PATH = r''# Replace with your actual path
IMAGE_SIZE = 500
NUM_CLASSES = 13 # bB, bK, bN, bP, bQ, bR, empty, wB, wK, wN, wP, wQ, wR
WARPED_BOARD_SIZE = 4000 # Size of the board after perspective warp, ensure cell_size is 500

# --- Classes (from your training script) ---
ALL_CLASSES = sorted(['bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'empty', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR'])
idx_to_class = {i: cls_name for i, cls_name in enumerate(ALL_CLASSES)}

# --- Device Configuration ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Transformation for Prediction (similar to 'val' or 'test' in your training) ---
data_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE + 32), # Ensure consistency with training
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path, num_classes, device):
    """Loads the trained EfficientNet model."""
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes) # Set pretrained=False if loading custom weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model state_dict from checkpoint: {model_path}")
        elif 'state_dict' in checkpoint:
             model.load_state_dict(checkpoint['state_dict'])
             print(f"Loaded model state_dict from checkpoint: {model_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model state_dict directly from: {model_path}")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load as if it's just the model's state_dict...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Successfully loaded model state_dict directly from: {model_path}")
        except Exception as e2:
            print(f"Second attempt to load model failed: {e2}")
            return None

    model = model.to(device)
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully and set to evaluation mode.")
    return model

def get_aruco_corners(image):
    """Detects ArUco markers and returns the inner corners of the chessboard."""
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    try:
        aruco_parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray_img)
    except AttributeError: # For older OpenCV versions
        print("Using legacy ArUco detection method.")
        parameters = cv2.aruco.DetectorParameters_create() # Fix: cv2.aruco
        marker_corners, marker_ids, _ = aruco.detectMarkers(gray_img, aruco_dictionary, parameters=parameters)

    if marker_ids is None or len(marker_ids) < 4:
        detected_count = len(marker_ids) if marker_ids is not None else 0
        print(f"Error: Less than 4 ArUco markers detected. Found {detected_count}.")
        return None

    detected_markers_map = {marker_id[0]: corner[0] for marker_id, corner in zip(marker_ids, marker_corners)}
    required_ids = [0, 1, 2, 3] 

    for req_id in required_ids:
        if req_id not in detected_markers_map:
            print(f"Error: Required ArUco marker ID {req_id} not found.")
            return None
    try:
        board_top_left = detected_markers_map[0][2]
        board_top_right = detected_markers_map[1][3]
        board_bottom_right = detected_markers_map[2][0]
        board_bottom_left = detected_markers_map[3][1]

    except IndexError:
        print(f"Error: Could not extract necessary marker corners.")
        return None
    except KeyError as e:
        print(f"Error: Critical marker ID {e} missing.")
        return None

    return np.array([board_top_left, board_top_right, board_bottom_right, board_bottom_left], dtype="float32")

def warp_and_crop_board(image, corners, warped_size):
    """Warps the perspective of the chessboard and returns the warped image."""
    destination_points = np.array([
        [0, 0], [warped_size - 1, 0],
        [warped_size - 1, warped_size - 1], [0, warped_size - 1]
    ], dtype="float32")
    perspective_matrix = cv2.getPerspectiveTransform(corners, destination_points)
    warped_chessboard_img = cv2.warpPerspective(image, perspective_matrix, (warped_size, warped_size))
    return warped_chessboard_img

def predict_cell(cell_image_pil, model, transform, device, idx_to_class_map):
    """Predicts the chess piece in a single cell image."""
    transformed_image = transform(cell_image_pil).unsqueeze(0) 
    transformed_image = transformed_image.to(device)
    with torch.no_grad():
        outputs = model(transformed_image)
        _, preds = torch.max(outputs, 1)
    predicted_class_idx = preds.item()
    return idx_to_class_map[predicted_class_idx]

def board_to_fen_piece_placement(board_array):
    """Converts an 8x8 board array (with piece codes) to FEN piece placement string."""
    fen = ""
    for row_idx in range(8):
        empty_count = 0
        for col_idx in range(8):
            piece = board_array[row_idx][col_idx] 
            
            fen_char = ''
            if piece == 'empty':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                
                color = piece[0]
                piece_type = piece[1]
                if color == 'w':
                    fen_char = piece_type.upper()
                elif color == 'b':
                    fen_char = piece_type.lower()
                fen += fen_char
                
        if empty_count > 0:
            fen += str(empty_count)
        if row_idx < 7:
            fen += "/" # Standard FEN uses '/' as rank separator
    return fen


def process_and_predict(image_path, model, data_transform, device, idx_to_class_map, warped_board_size):
    """
    Processes a full chessboard image, predicts pieces, and returns FEN.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}.")
        return None, None

    board_corners = get_aruco_corners(img)
    if board_corners is None:
        print("Could not find ArUco corners. Aborting.")
        return None, None

    warped_board = warp_and_crop_board(img, board_corners, warped_board_size)
    
    cell_size_on_warped = warped_board_size // 8
    
    board_state_array = [['' for _ in range(8)] for _ in range(8)] 

    print("Processing and predicting cells...")
    for r in range(8): 
        for f in range(8): 
            start_y, end_y = r * cell_size_on_warped, (r + 1) * cell_size_on_warped
            start_x, end_x = f * cell_size_on_warped, (f + 1) * cell_size_on_warped
            
            cell_bgr = warped_board[start_y:end_y, start_x:end_x]
            cell_rgb = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGB)
            cell_pil = Image.fromarray(cell_rgb)

            predicted_piece_code = predict_cell(cell_pil, model, data_transform, device, idx_to_class_map)
            board_state_array[r][f] = predicted_piece_code

    piece_placement_fen = board_to_fen_piece_placement(board_state_array)
    
    active_color = "w"
    castling_availability = "KQkq" 
    en_passant_target = "-"
    halfmove_clock = "0"
    fullmove_number = "1"
    
    full_fen = f"{piece_placement_fen} {active_color} {castling_availability} {en_passant_target} {halfmove_clock} {fullmove_number}"
    
    return full_fen, board_state_array


if __name__ == "__main__":
    chessboard_image_path = r"" # <--- CHANGE THIS TO YOUR INPUT IMAGE

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()
    if not os.path.exists(chessboard_image_path):
        print(f"Error: Input image file not found at {chessboard_image_path}")
        exit()

    trained_model = load_model(MODEL_PATH, NUM_CLASSES, device)

    if trained_model:
        print(f"\nProcessing chessboard image: {chessboard_image_path}")
        fen_output, board_array = process_and_predict(
            chessboard_image_path,
            trained_model,
            data_transform,
            device,
            idx_to_class,
            WARPED_BOARD_SIZE
        )

        if fen_output:
            print("\nPredicted Board Array (internal representation):")
            for row in board_array:
                print(row)
            print(f"\nPredicted FEN: {fen_output}")