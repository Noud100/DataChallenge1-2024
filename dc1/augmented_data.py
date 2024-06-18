import torch
from torchvision import transforms as v2
import matplotlib.pyplot as plt
from dc1.image_dataset import ImageDataset
from pathlib import Path
import numpy as np

train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

# Checking how many images each class has
classes, counts = np.unique(train_dataset.targets, return_counts=True)
counts_dict = dict(zip(classes, counts))
print(counts_dict)

# To see the result of these transforms, check data_augmentation.py
transforms_final = v2.Compose([
    v2.RandomSolarize(128, p=0.2),
    v2.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(1, 1.2), shear=(-8, 8)),
    v2.RandomPerspective(distortion_scale=0.6, p=0.2),
    v2.RandomRotation(degrees=(-5, 5))
])

# Initialize lists to hold augmented images and their labels
augmented_images = []
augmented_labels = []

for class_idx in range(0, 6):  # Classes are labeled from 0 to 5
    class_images = [img for img, label in zip(train_dataset.imgs, train_dataset.targets) if label == class_idx]

    # first check how many images we need to make of the class
    total_images = 20000
    if counts_dict[class_idx] < total_images:
        index = 0
        for i in range(total_images - counts_dict[class_idx]):
            if index < counts[class_idx]:
                img = class_images[index]
                transformed_img = transforms_final(torch.from_numpy(img))
                augmented_images.append(transformed_img.numpy())
                augmented_labels.append(class_idx)
                index += 1
            else:
                index = 0
                img = class_images[index]
                transformed_img = transforms_final(torch.from_numpy(img))
                augmented_images.append(transformed_img.numpy())
                augmented_labels.append(class_idx)
    print(f'Class {class_idx} done!')

# At this point, `augmented_images` and `augmented_labels` does not contain the original images and labels

# Checking how many images each class has
classes, counts = np.unique(augmented_labels, return_counts=True)
counts_aug_dict = dict(zip(classes, counts))
print(counts_aug_dict)
print(len(augmented_images))


# Plot the first 10 augmented images
for i in range(len(augmented_images[:10])):
    plt.imshow(augmented_images[i].reshape(128, 128), cmap='gray')
    plt.title(f'Image of class: {augmented_labels[i]}')
    plt.show()

# Save the augmented data as numpy arrays (uncomment if you do not have these files yet!)
np.save('augmented_targets_v5', np.array(augmented_labels))
np.save('augmented_imgs_v5', np.array(augmented_images))
