import os
import cv2
import albumentations as A
import shutil

# Define the path to your original dataset and where augmented data will be saved
original_data_dir = r'' # Replace with your actual path
augmented_data_dir = r''# Replace with your actual path

# Define the 13 classes based on your folder names
classes = ['bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'empty', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR']

# Number of augmented versions to create per original image (target is 10x, so 9 new versions + original)
num_augmented_samples_per_original = 9 # We want 9 new images to make a total of 10 (including original)

# Define the augmentation pipeline
# We'll create a few different augmentation strategies to get variety
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=[0,0,0]),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
        A.GaussianBlur(blur_limit=5, p=1.0)
    ], p=0.5),
    A.Perspective(scale=(0.01, 0.05), p=0.3),
    A.CoarseDropout(max_holes=8, max_height=25, max_width=25, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.3) # Simulates occlusion
])


def augment_and_save_images():
    if not os.path.exists(augmented_data_dir):
        os.makedirs(augmented_data_dir)
        print(f"Created directory: {augmented_data_dir}")

    for class_name in classes:
        original_class_path = os.path.join(original_data_dir, class_name)
        augmented_class_path = os.path.join(augmented_data_dir, class_name)

        if not os.path.exists(augmented_class_path):
            os.makedirs(augmented_class_path)
            print(f"Created directory: {augmented_class_path}")

        if not os.path.exists(original_class_path):
            print(f"Warning: Original class directory not found: {original_class_path}")
            continue

        print(f"Processing class: {class_name}")
        image_files = [f for f in os.listdir(original_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name_original in image_files:
            img_path = os.path.join(original_class_path, img_name_original)
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read image {img_path}. Skipping.")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations expects RGB

                # Save the original image
                base_name, ext = os.path.splitext(img_name_original)
                original_save_path = os.path.join(augmented_class_path, f"{base_name}_orig{ext}")
                cv2.imwrite(original_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                # Generate and save augmented images
                for i in range(num_augmented_samples_per_original):
                    augmented = transform(image=image)
                    augmented_image = augmented['image']
                    
                    # Construct new filename
                    new_img_name = f"{base_name}_aug_{i+1}{ext}"
                    save_path = os.path.join(augmented_class_path, new_img_name)
                    cv2.imwrite(save_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
                    
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    print("Data augmentation complete.")
    # Verify counts
    for class_name in classes:
        augmented_class_path = os.path.join(augmented_data_dir, class_name)
        if os.path.exists(augmented_class_path):
            count = len([name for name in os.listdir(augmented_class_path) if os.path.isfile(os.path.join(augmented_class_path, name))])
            print(f"Class {class_name} has {count} images after augmentation.")

if __name__ == '__main__':
    # IMPORTANT: Make sure your 'final_original_dataset' directory is in the same
    # location as this script, or provide the full absolute path.
    # Example: original_data_dir = r'C:\path\to\your\final_original_dataset'
    # Example: augmented_data_dir = r'C:\path\to\your\final_augmented_dataset'
    
    augment_and_save_images()