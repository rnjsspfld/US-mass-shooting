import pandas as pd
import requests
import time
import random
import os
import matplotlib.pyplot as plt

# df = pd.read_csv('/Users/hyerinkwon/Desktop/Research Group/SMAD_mass_shooting/SMAD_MS/dedup_img.csv')
#
# # Assuming x contains the index numbers
# for x, url in zip(df['x'], df['media_url']):
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             # Save the image with the name img{index}.jpg
#             filename = f"/Users/hyerinkwon/Desktop/Research Group/SMAD_mass_shooting/SMAD_MS/tt_dedup_image/img{x}.jpg"
#             with open(filename, 'wb') as f:
#                 f.write(response.content)
#             print(f"Image {x} downloaded successfully.")
#         else:
#             print(f"Failed to download image {x}. Status code: {response.status_code}")
#
#         # Add a random time interval between 0 and 1 seconds
#         random_sec = random.uniform(0, 1)
#         time.sleep(random_sec)
#     except Exception as e:
#         print(f"Error downloading image {x}: {str(e)}")


# image_files = []
#
# # Process files in batches
# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
#             image_files.append(file)
#
# print(len(image_files))

import tensorflow as tf


import tensorflow as tf
import numpy as np
import os

# Path to your local folder containing images
folder_path = "/Users/hyerinkwon/Desktop/Research Group/SMAD_mass_shooting/SMAD_MS/tt_dedup_image"

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Path to your local folder containing images
folder_path = "/Users/hyerinkwon/Desktop/Research Group/SMAD_mass_shooting/SMAD_MS/tt_dedup_image"

# Define a transformation to apply to each image
# In this case, we only convert the image to a PyTorch tensor
transform = transforms.Compose([
    transforms.CenterCrop(1200), ## img_size.py 통해, biggest height, width 1200 알게됨
    transforms.ToTensor(),
])

# Create a custom dataset class to load images
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.file_paths = [os.path.join(self.folder_path, filename) for filename in os.listdir(self.folder_path)]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img = Image.open(img_path).convert('RGB')  # Ensure RGB mode
        if self.transform:
            img = self.transform(img)
        filename = os.path.basename(img_path)
        return img, filename

# def custom_collate(batch): ## for padding to maintain original image... but not ideal...
#     images = [img for img in batch]
#     max_width = max([img.shape[2] for img in images])
#     max_height = max([img.shape[1] for img in images])
#     padded_images = []
#     for img in images:
#         pad_width = max_width - img.shape[2]
#         pad_height = max_height - img.shape[1]
#         padded_img = torch.nn.functional.pad(img, (0, pad_width, 0, pad_height))
#         padded_images.append(padded_img)
#     return torch.stack(padded_images, 0)


# Create an instance of the custom dataset
dataset = CustomDataset(folder_path, transform=transform)

# Create a DataLoader to efficiently load and iterate over the dataset
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #collate_fn=custom_collate)

# Iterate over the dataloader and display the first batch of images
for batch in dataloader:
    images, filenames = batch
    for image, filename in zip(images, filenames):
        # Convert the PyTorch tensor to a numpy array and display the image
        image_np = image.permute(1, 2, 0).numpy()  # Convert (C, H, W) to (H, W, C) for display
        plt.imshow(image_np)
        plt.title(filename)
        plt.axis('off')
        plt.show()
    break  # Only display the first batch


