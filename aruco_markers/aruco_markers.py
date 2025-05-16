import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Directory to save the markers
output_dir = r"" #Update this path
os.makedirs(output_dir, exist_ok=True)

# Use 6x6 dictionary (250 IDs)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_size = 200  # pixels

# Generate and save markers for IDs 0–3
for marker_id in range(4):
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    file_path = os.path.join(output_dir, f"marker_{marker_id}.png")
    cv2.imwrite(file_path, marker_image)
    print(f"Saved: {file_path}")

    # Optional: show the marker
    plt.figure()
    plt.imshow(marker_image, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(f'ArUco Marker {marker_id}')
    plt.show()