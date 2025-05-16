import os
import random
import shutil

def process_chess_images_numbered_output(main_folder_path, output_folder_path):
    """
    Processes chess images by selecting one random image from each subfolder
    within the main FEN-named folders and copying it to a corresponding subfolder
    in the output directory. Output images are numbered sequentially within
    each piece subfolder.

    Args:
        main_folder_path (str): The path to the main directory containing
                                 the 100 FEN-named folders.
        output_folder_path (str): The path to the directory where the
                                   randomly selected images will be saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Created output directory: {output_folder_path}")

    # Get the list of all FEN-named main folders
    try:
        fen_folders = [
            folder for folder in os.listdir(main_folder_path)
            if os.path.isdir(os.path.join(main_folder_path, folder))
        ]
    except FileNotFoundError:
        print(f"Error: Main folder not found at {main_folder_path}")
        return
    except Exception as e:
        print(f"An error occurred while accessing the main folder: {e}")
        return

    if not fen_folders:
        print(f"No FEN folders found in {main_folder_path}")
        return

    print(f"Found {len(fen_folders)} FEN folders in {main_folder_path}.")

    # Define the expected subfolder names (piece types)
    piece_subfolder_names = [
        "bB", "bK", "bN", "bP", "bQ", "bR",
        "wB", "wK", "wN", "wP", "wQ", "wR",
        "empty"
    ]

    # Initialize counters for each piece type for output file naming
    output_file_counters = {piece_name: 1 for piece_name in piece_subfolder_names}

    # Create subfolders in the output directory
    for piece_name in piece_subfolder_names:
        output_piece_path = os.path.join(output_folder_path, piece_name)
        if not os.path.exists(output_piece_path):
            os.makedirs(output_piece_path)

    total_images_copied = 0
    processed_fen_folders_count = 0

    # Iterate through each FEN-named main folder
    for fen_folder_name in fen_folders:
        fen_folder_path = os.path.join(main_folder_path, fen_folder_name)
        print(f"\nProcessing FEN folder: {fen_folder_name}")
        processed_fen_folders_count += 1

        # Iterate through each piece subfolder type
        for piece_name in piece_subfolder_names:
            current_piece_subfolder_path = os.path.join(fen_folder_path, piece_name)

            if not os.path.isdir(current_piece_subfolder_path):
                print(f"  Warning: Subfolder '{piece_name}' not found in '{fen_folder_name}'. Skipping.")
                continue

            try:
                images_in_subfolder = [
                    img for img in os.listdir(current_piece_subfolder_path)
                    if os.path.isfile(os.path.join(current_piece_subfolder_path, img))
                ]
            except Exception as e:
                print(f"  Error accessing images in {current_piece_subfolder_path}: {e}")
                continue

            if not images_in_subfolder:
                print(f"  Warning: No images found in '{current_piece_subfolder_path}'. Skipping.")
                continue

            # Select one random image
            try:
                random_image_name = random.choice(images_in_subfolder)
            except IndexError:
                print(f"  Warning: Could not select a random image from '{current_piece_subfolder_path}' (empty list after filtering). Skipping.")
                continue

            source_image_path = os.path.join(current_piece_subfolder_path, random_image_name)
            
            # Determine the file extension
            _, ext = os.path.splitext(random_image_name)
            if not ext: # Handle cases where files might not have extensions (though images usually do)
                ext = ".png" # Default to .png or handle as an error
                print(f"  Warning: Image '{random_image_name}' in '{current_piece_subfolder_path}' has no extension. Assuming '{ext}'.")


            # Define the output path and numbered filename
            output_piece_folder_for_type = os.path.join(output_folder_path, piece_name)
            
            # Get the current counter for this piece type
            file_number = output_file_counters[piece_name]
            numbered_image_name = f"{file_number}{ext}"
            destination_image_path = os.path.join(output_piece_folder_for_type, numbered_image_name)

            # Copy the selected image
            try:
                shutil.copy2(source_image_path, destination_image_path)
                print(f"  Copied '{random_image_name}' from '{piece_name}' to '{output_piece_folder_for_type}' as '{numbered_image_name}'")
                total_images_copied += 1
                # Increment the counter for the next image of this piece type
                output_file_counters[piece_name] += 1
            except Exception as e:
                print(f"  Error copying image {source_image_path} to {destination_image_path}: {e}")

    print(f"\n--- Processing Complete ---")
    print(f"Processed {processed_fen_folders_count} FEN folders.")
    print(f"Total images copied: {total_images_copied}")

    # Verification: Check if each output piece folder has the expected number of images (should be number of FEN folders)
    expected_images_per_piece_folder = len(fen_folders)
    for piece_name in piece_subfolder_names:
        output_piece_dir = os.path.join(output_folder_path, piece_name)
        if os.path.exists(output_piece_dir):
            num_files = len([f for f in os.listdir(output_piece_dir) if os.path.isfile(os.path.join(output_piece_dir, f))])
            if num_files != expected_images_per_piece_folder:
                print(f"  Verification Warning: Output folder '{piece_name}' contains {num_files} images, expected {expected_images_per_piece_folder}.")
            else:
                print(f"  Verification OK: Output folder '{piece_name}' contains {num_files} images.")
        else:
             print(f"  Verification Warning: Output folder '{piece_name}' was not created or is missing.")


    expected_total_images = len(fen_folders) * len(piece_subfolder_names)
    if total_images_copied != expected_total_images:
        print(f"Warning: Expected to copy {expected_total_images} images, but copied {total_images_copied}. "
              "This might be due to missing subfolders or images in the source.")

if __name__ == "__main__":
    # --- Configuration ---
    # !!! IMPORTANT: Update these paths before running the script !!!
    
    # Path to your main folder containing the 100 FEN folders.
    # Example for Windows: r"C:\Users\YourUser\Desktop\Chess_Dataset\split_images"
    # Example for Linux/macOS: "/home/user/datasets/chess_raw/split_images"
    # Based on your images, it looks like "split_images" is the target, or its parent.
    main_dataset_folder = r""# Replace with your actual path 

    # Path where the script will create subfolders (bB, bK, etc.)
    # and save the randomly selected, numbered images.
    # Example for Windows: r"C:\Users\YourUser\Desktop\Chess_Dataset\processed_numbered_output"
    # Example for Linux/macOS: "/home/user/datasets/chess_processed_numbered"
    output_dataset_folder = r""# Replace with your actual path

    # --- Validate paths ---
    if main_dataset_folder == r"PATH_TO_YOUR_MAIN_DATASET_FOLDER" or \
       output_dataset_folder == r"PATH_TO_YOUR_OUTPUT_FOLDER" or \
       not main_dataset_folder or not output_dataset_folder: # Also check for empty strings
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Please update 'main_dataset_folder' and         !!!")
        print("!!!        'output_dataset_folder' variables in the script !!!")
        print("!!!        with the correct paths to your data.             !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print(f"Starting image processing...")
        print(f"Main dataset folder: {main_dataset_folder}")
        print(f"Output dataset folder: {output_dataset_folder}")
        process_chess_images_numbered_output(main_dataset_folder, output_dataset_folder)