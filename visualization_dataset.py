import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def get_all_image_paths(base_folder):
    # Collect all image paths from the subfolders
    image_paths = []
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    relative_path = os.path.join(folder_name, file_name)
                    image_paths.append(relative_path)
    return image_paths

def plot_selected_images(base_folder, selected_paths):
    # Plotting
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))  # Adjusted grid size for 16 images
    axes = axes.ravel()  # Flatten the array of axes for easy iteration

    for ax, relative_path in zip(axes, selected_paths):
        image_path = os.path.join(base_folder, relative_path)
        img = Image.open(image_path)
        # Ensure grayscale images stay grayscale and RGB stay RGB
        if img.mode != 'RGB':
            img = img.convert('L')  # Convert to grayscale if not already
            ax.imshow(np.array(img), cmap='gray')  # Display using a grayscale colormap
        else:
            ax.imshow(img)
        ax.set_xticks([])  # Remove x-axis markers
        ax.set_yticks([])  # Remove y-axis markers

    plt.tight_layout()
    plt.show()

base_folders = ['Datasets/RAF-FER-SFEW-AN', 'Datasets/combined_dataset_processed_128_1']

# Get all image paths from both folders
image_paths_1 = get_all_image_paths(base_folders[0])
image_paths_2 = get_all_image_paths(base_folders[1])

# Find common images
common_paths = list(set(image_paths_1) & set(image_paths_2))

# Ensure there are enough common images to select from
num_images_to_select = 16
if len(common_paths) < num_images_to_select:
    raise ValueError("Not enough common images in the datasets to select the desired number of images.")

# Randomly select from the common images
selected_paths = random.sample(common_paths, num_images_to_select)

# Plot selected images from each folder using common paths
for base_folder in base_folders:
    print(f"Displaying images from: {base_folder}")
    plot_selected_images(base_folder, selected_paths)
