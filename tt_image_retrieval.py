import os

from skimage.io import imread
from skimage.transform import resize
import numpy as np

# prepare data
import os
import gc
from skimage.io import imread
import numpy as np
from memory_profiler import profile

folder_path = "/Users/hyerinkwon/Desktop/Research Group/mass_shooting/SMAD_MS/tt_image2"  # Replace with your actual directory
batch_size = 50  # Adjust batch size as needed
start_idx = 110000
end_idx = 115000
drive_path = "/Users/hyerinkwon/Desktop/Research Group/mass_shooting/SMAD_MS/data"
# Create an empty list to store all processed images (in their original sizes)
all_images = []

for i in range(start_idx, end_idx, batch_size):
    images1 = []  # Temporary list for current batch

    for file in os.listdir(folder_path)[i:i+batch_size]:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, file)
            try:
                # Load the image without resizing
                image = imread(image_path)
                images1.append(image)
            except Exception as e:
                print(f"Error loading image '{file}': {e}")

    # Append processed images to all_images
    all_images.extend(images1)

    # Clear memory after processing each batch
    del images1[:]
    gc.collect()

    print(f"Processed batch {i // batch_size + 1}/{end_idx // batch_size}")

# Store all processed images as a NumPy array (optional)
if all_images:
    # Convert the list of images to a NumPy array (may require adjustments for sizes)
    all_images_array = np.array(all_images, dtype=object) ## dtype=object 가 key 였어!!!! 아오 짜잉나!!!
    save_path = os.path.join(drive_path, "tt_image2_23.npy")
    np.save(save_path, all_images_array)
    print(f"Saved all images as NumPy array to {save_path}")
else:
    print("No images were processed. Check your folder path and file formats.")
