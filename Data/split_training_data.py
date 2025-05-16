import os
import random
from sklearn.model_selection import train_test_split

# Path to your augmented dataset (where the actual image files are)
augmented_data_dir = r''# Replace with your actual path

# Path to the new directory where text files will be stored
index_files_dir = r''# Replace with your actual path

# Define the split ratios
train_ratio = 0.80
val_ratio = 0.10
test_ratio = 0.10 # train_ratio + val_ratio + test_ratio should be 1.0

# Classes
classes = ['bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'empty', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR']

def create_index_files():
    if os.path.exists(index_files_dir):
        print(f"Index files directory '{index_files_dir}' already exists. Please remove it or choose a new name to avoid appending to existing files.")
        # To automatically remove:
        # import shutil
        # shutil.rmtree(index_files_dir)
        # print(f"Removed existing directory: {index_files_dir}")
        # os.makedirs(index_files_dir)
        return # Exit if directory exists to prevent accidental overwrite/append

    os.makedirs(index_files_dir, exist_ok=True)
    print(f"Created directory for index files: {index_files_dir}")

    # Create subdirectories for train, val, test within the index_files_dir
    # to keep the text files organized, though not strictly necessary for this script.
    for split_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(index_files_dir, split_name), exist_ok=True)

    # Get the absolute path to the augmented data directory
    # This ensures paths in text files are absolute, making them more robust
    abs_augmented_data_dir = os.path.abspath(augmented_data_dir)

    for class_name in classes:
        class_source_dir = os.path.join(abs_augmented_data_dir, class_name)

        if not os.path.exists(class_source_dir):
            print(f"Warning: Source directory for class {class_name} not found at {class_source_dir}. Skipping.")
            continue

        # Get a list of image filenames (not full paths yet)
        image_filenames = [f for f in os.listdir(class_source_dir) if os.path.isfile(os.path.join(class_source_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_filenames) # Shuffle images for random splitting

        if not image_filenames:
            print(f"Warning: No images found in {class_source_dir} for class {class_name}.")
            continue

        # First, split into training and (validation + test)
        train_filenames, temp_filenames = train_test_split(image_filenames, test_size=(val_ratio + test_ratio), random_state=42, shuffle=True)

        # Now split (validation + test) into validation and test
        if (val_ratio + test_ratio) == 0:
            val_filenames = []
            test_filenames = temp_filenames
        else:
            relative_test_size = test_ratio / (val_ratio + test_ratio)
            if len(temp_filenames) > 1:
                 val_filenames, test_filenames = train_test_split(temp_filenames, test_size=relative_test_size, random_state=42, shuffle=True)
            elif len(temp_filenames) == 1 and relative_test_size < 1.0:
                val_filenames = temp_filenames
                test_filenames = []
            elif len(temp_filenames) == 1 and relative_test_size >= 1.0:
                val_filenames = []
                test_filenames = temp_filenames
            else:
                val_filenames = []
                test_filenames = []

        print(f"Class {class_name}: Total {len(image_filenames)}, Train {len(train_filenames)}, Val {len(val_filenames)}, Test {len(test_filenames)}")

        # Function to write paths to a text file
        def write_paths_to_file(filenames_list, split_name_str, current_class_name):
            # Construct filename e.g., train_bB.txt, val_empty.txt
            txt_filename = f"{split_name_str}_{current_class_name}.txt"
            # Store these files in subfolders like 'index_files_dir/train/train_bB.txt'
            txt_filepath = os.path.join(index_files_dir, split_name_str, txt_filename)

            with open(txt_filepath, 'w') as f:
                for img_filename in filenames_list:
                    # Construct the absolute path to the image
                    abs_img_path = os.path.join(class_source_dir, img_filename)
                    f.write(f"{abs_img_path}\n")
            print(f"  Created index file: {txt_filepath}")

        write_paths_to_file(train_filenames, 'train', class_name)
        write_paths_to_file(val_filenames, 'val', class_name)
        write_paths_to_file(test_filenames, 'test', class_name)

    print("\nIndex file creation complete.")
    # Verification
    for split_name in ['train', 'val', 'test']:
        print(f"\nVerifying files in {os.path.join(index_files_dir, split_name)}:")
        count_files = 0
        total_lines = 0
        for fname in os.listdir(os.path.join(index_files_dir, split_name)):
            if fname.endswith(".txt"):
                count_files +=1
                with open(os.path.join(index_files_dir, split_name, fname), 'r') as f_read:
                    lines_in_file = len(f_read.readlines())
                    print(f"  File {fname} has {lines_in_file} entries.")
                    total_lines += lines_in_file
        print(f"Total .txt files in {split_name}: {count_files}, Total image paths listed: {total_lines}")


if __name__ == '__main__':
    # IMPORTANT: Make sure your 'final_augmented_dataset' directory is in the same
    # location as this script, or provide the full absolute path.
    # Example: augmented_data_dir = r'C:\path\to\your\final_augmented_dataset'
    # Example: index_files_dir = r'C:\path\to\your\chess_pieces_index_files'

    random.seed(42) # For reproducible splits
    create_index_files()