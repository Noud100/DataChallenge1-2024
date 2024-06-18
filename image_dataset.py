import numpy as np
import torch
import requests
import io
from os import path
from typing import Tuple
from pathlib import Path
import os
import cv2


def contrast_stretching(image):
    # Calculate minimum and maximum pixel values excluding the lowest 5% to avoid extreme outliers
    min_val = np.percentile(image, 15)
    max_val = np.percentile(image, 80)

    # Scale pixel values to span the entire intensity range (0 to 255)
    stretched_image = (image - min_val) * (255.0 / (max_val - min_val))

    # Ensure pixel values are within the valid range [0, 255]
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)

    return stretched_image


class ImageDataset:
    """
    Creates a DataSet from numpy arrays while keeping the data
    in the more efficient numpy arrays for as long as possible and only
    converting to torch tensors when needed (torch tensors are the objects used
    to pass the data through the neural network and apply weights).
    """

    def __init__(self, x: Path, y: Path) -> None:
        # Set true or false if you want to use augmented data and contrasted data
        augmentation = False
        contrasting = False
        preprocessed_augmentation = False

        if not augmentation and not preprocessed_augmentation:
            # When using no augmented data
            # Target labels
            self.targets = ImageDataset.load_numpy_arr_from_npy(y)
            # Images
            self.imgs = ImageDataset.load_numpy_arr_from_npy(x)
            print('Note: You are not using augmented data')

        if preprocessed_augmentation:
            # Target labels
            self.targets = np.append(ImageDataset.load_numpy_arr_from_npy(y), np.load("preprocessed_augmented_targets_2.npy"))
            # Images
            self.imgs = np.append(ImageDataset.load_numpy_arr_from_npy(x), np.load("preprocessed_augmented_imgs_2.npy"))
            print('Note: You are using augmented data with 3 preprocessing techniques')


        if augmentation:
            if 'train' in str(x):
                # When using augmented data
                # Target labels
                self.targets = np.append(ImageDataset.load_numpy_arr_from_npy(y), np.load("augmented_targets_v5.npy"))
                # # Images
                self.imgs = np.concatenate([ImageDataset.load_numpy_arr_from_npy(x), np.load("augmented_imgs_v5.npy")])
                print('Note: You are using augmented data')
            else: # Target labels
                self.targets = ImageDataset.load_numpy_arr_from_npy(y)
                # Images
                self.imgs = ImageDataset.load_numpy_arr_from_npy(x)

        if contrasting:
            for i in range(len(self.imgs)):
                self.imgs[i] = contrast_stretching(self.imgs[i]) # Contrast stretching the images (normalizing)
                img = self.imgs[i]
                img_uint8 = img.astype(np.uint8)  # Converting to uint8 images so that we can use cv2

                # Reshape the array to 2D
                img_2d = img_uint8.reshape(128, 128)

                # Histogram equalization
                equalized_image_2d = cv2.equalizeHist(img_2d)

                # Reshape back to original shape
                equalized_img = equalized_image_2d.reshape(1, 128, 128)
                self.imgs[i] = equalized_img

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        image = torch.from_numpy(self.imgs[idx] / 255).float()
        label = self.targets[idx]
        return image, label

    @staticmethod
    def load_numpy_arr_from_npy(path: Path) -> np.ndarray:
        """
        Loads a numpy array from local storage.

        Input:
        path: local path of file

        Outputs:
        dataset: numpy array with input features or labels
        """

        return np.load(path)


def load_numpy_arr_from_url(url: str) -> np.ndarray:
    """
    Loads a numpy array from surfdrive.

    Input:
    url: Download link of dataset

    Outputs:
    dataset: numpy array with input features or labels
    """

    response = requests.get(url)
    response.raise_for_status()

    return np.load(io.BytesIO(response.content))


if __name__ == "__main__":
    cwd = os.getcwd()
    if path.exists(path.join(cwd + "data/")):
        print("Data directory exists, files may be overwritten!")
    else:
        os.mkdir(path.join(cwd, "../../data/"))
    ### Load labels
    train_y = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/i6MvQ8nqoiQ9Tci/download"
    )
    np.save("../../data/Y_train.npy", train_y)
    test_y = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/wLXiOjVAW4AWlXY/download"
    )
    np.save("../../data/Y_test.npy", test_y)
    ### Load data
    train_x = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/4rwSf9SYO1ydGtK/download"
    )
    np.save("../../data/X_train.npy", train_x)
    test_x = load_numpy_arr_from_url(
        url="https://surfdrive.surf.nl/files/index.php/s/dvY2LpvFo6dHef0/download"
    )
    np.save("../../data/X_test.npy", test_x)
