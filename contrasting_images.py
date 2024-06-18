import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dc1.image_dataset import ImageDataset
import torch
from pathlib import Path
from PIL import Image
import cv2

train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

img = train_dataset.imgs[0]

# plotting the first picture of the train data
plt.imshow(img.reshape(128, 128), cmap='gray')
plt.title('Original image')
plt.show()

# Assuming your image array is named 'image_array'
# Convert the array to uint8 format
image_array_uint8 = img.astype(np.uint8)

# Reshape the array to 2D
image_2d = image_array_uint8.reshape(128, 128)

# Perform histogram equalization
equalized_image_2d = cv2.equalizeHist(image_2d)

# Reshape back to original shape
equalized_img = equalized_image_2d.reshape(1, 128, 128)

# plotting the first picture of the train data
plt.imshow(equalized_img.reshape(128, 128), cmap='gray')
plt.title('Equalized image')
plt.show()


def contrast_stretching(image):
    # Calculate minimum and maximum pixel values excluding the lowest 5% to avoid extreme outliers
    min_val = np.percentile(image, 15)
    max_val = np.percentile(image, 80)

    # Scale pixel values to span the entire intensity range (0 to 255)
    stretched_image = (image - min_val) * (255.0 / (max_val - min_val))

    # Ensure pixel values are within the valid range [0, 255]
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)

    return stretched_image


# Apply contrast stretching
stretched_img = contrast_stretching(img)

# plotting the first picture of the train data
plt.imshow(contrast_stretching(img).reshape(128, 128), cmap='gray')
plt.title('Stretched image')
plt.show()

# plotting the first picture of the train data
plt.imshow(contrast_stretching(equalized_img).reshape(128, 128), cmap='gray')
plt.title('Stretched and Equalized image')
plt.show()

for i in range(len(train_dataset.imgs[:10])):
    print(i)

