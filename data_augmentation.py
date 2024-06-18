import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dc1.image_dataset import ImageDataset
from pathlib import Path
import tensorflow as tf
import torch
from torchvision.transforms import v2



# loading the data
train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

# Testing Data Augmentation with tensorflow
# testing on the first image of the train data
first_image = train_dataset.imgs[0]
flipped_first_image = tf.image.flip_left_right(first_image)

# Plotting the original image
plt.imshow(first_image.reshape(128, 128), cmap='gray')
plt.title(f'First image (class: {train_dataset.targets[0]})')
plt.show()

# Plotting the flipped image
plt.imshow(flipped_first_image.numpy().reshape(128, 128), cmap='gray')
plt.title(f'First image flipped (class: {train_dataset.targets[0]})')
plt.show()

# Notes on TensorFlow data augmentation: Flipping the image is just one way of using data augmentation. There are many
# other possibilities: giving the image a small tilt, bigger color contrasts, cropping the image, etc.
# We need to find out if it is possible to flip the images in our case. For example: what if one decease only appears
# in your left lung? Then flipping the image might not make it easier for the model to train.
# One more note: It seems like that Pytorch is able to transform and augment images itself. This is something that we
# need to find out for the next sprint, as for now we just do research and some (very basic) testing regarding data
# augmentation during sprint 1.
# Source Pytorch: https://pytorch.org/vision/stable/transforms.html

print(type(train_dataset.imgs[0]))

# Testing with Pytorch data augmentation. Note: you need Torch 2.2.1


def visualize_augmented(augmented, augmentations):
    """
    Plots an augmented image
    :param augmented: augmented image we want to visualize
    :param augmentations: list of augmentations that is applied to the augmented image
    """
    plt.imshow(augmented.permute(1, 2, 0), cmap='gray')
    aug_str = ', '.join(augmentations)
    plt.title(f'Image with augmentations: {aug_str}')
    plt.show()


# We are going to start with some simple augmentations. We start with crop.
img = torch.from_numpy(first_image)
transforms_cropped = v2.Compose([
    v2.RandomResizedCrop(size=(128, 128), antialias=True)
])

visualize_augmented(transforms_cropped(img), ['cropped'])
# As you can see, the cropped augmentation might be tricky. Since the decease might not cover the whole X-ray, the
# decease might be cropped out of the image. Therefore, we need to look at other augmentations.

transforms_horflipped = v2.Compose([
    v2.RandomHorizontalFlip(p=1.0)
])
visualize_augmented(transforms_horflipped(img), ['horizontal flip'])
# It might be hard to notice on these X-rays, but the image here has been horizontally flipped (you can see that the
# water-mark is now in the top-right corner instead of the top left).
# We need to find out if it is possible to flip the images in our case. For example: what if one decease only appears
# in your left lung? Then flipping the image might not make it easier for the model to train.

transforms_normalized = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
visualize_augmented(transforms_normalized(img), ['normalized color'])
# What we did here is normalize the colors of the X-ray. That way, light and dark parts become more distinct.
# However, the deceases we have are hard to spot. The lungs are quite bright, so a decease in a lung might be 'filtered
# out' because of the normalized colors.

# We are also able to combine transforms (note: these are not the transforms we want to use, this is just to show an
# example).
transforms = v2.Compose([
    v2.RandomResizedCrop(size=(128, 128), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),  # note that p is the probability that the image is flipped
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
visualize_augmented(transforms(img), ['cropped', 'horizontal flip', 'normalized color'])

# What we need to do for now is selecting a couple of augmentations that we can apply to our data. See list for all
# augmentations here: https://pytorch.org/vision/stable/transforms.html (scroll a bit down)

transforms_vertflipped = v2.Compose([
    v2.RandomVerticalFlip(p=1.0)
    ])
visualize_augmented(transforms_vertflipped(img), ['vertical flip'])
# This augmentation is good I think.

transforms_rotation = v2.Compose([
    v2.RandomRotation(degrees=(0, 180))   # random rotation between 0 and 180 degrees
    ])
visualize_augmented(transforms_rotation(img), ['between 0 and 180 degrees rotation'])

transforms_resize = v2.Compose([
    v2.Resize(size=50)   # making the picture more blurry. The lower, the blurrier (100 is the original image)
    ])
visualize_augmented(transforms_resize(img), ['resize'])

transforms_perspective = v2.Compose([
    v2.RandomPerspective(distortion_scale=0.6, p=1.0) # performs random perspective changes
    ])
visualize_augmented(transforms_perspective(img), ['perspective transformer'])

transforms_affine = v2.Compose([
    v2.RandomAffine(degrees=(-15, 15),  # Random affine transformations
                    translate=(0.1, 0.1),
                    scale=(1, 1.2),
                    shear=(-8, 8))
])
visualize_augmented(transforms_affine(img), ['affine'])


# Will probably make the model perform worse since it can block out the desease
transforms_erasing = v2.Compose([
    v2.RandomErasing(p=0.5,  # Randomly erase rectangular blocks of pixels
                      scale=(0.02, 0.4),
                      ratio=(0.3, 4.0),
                      value=(128))
])

visualize_augmented(transforms_erasing(img), ['erasing'])

transforms_solarize = v2.Compose([
    v2.RandomSolarize(128, p=0.2) # Randomly invert pixel intensities in a portion of images
])

visualize_augmented(transforms_solarize(img), ['solarize'])

transforms_final = v2.Compose([
    v2.RandomSolarize(128, p=0.2),  # Randomly invert pixel intensities in a portion of images
    v2.RandomAffine(degrees=(-15, 15),  # Random affine transformations
                    translate=(0.1, 0.1),
                    scale=(1, 1.2),
                    shear=(-8, 8)),
    v2.RandomPerspective(distortion_scale=0.6, p=0.2),
    v2.RandomRotation(degrees=(-15, 15))
])

visualize_augmented(transforms_final(img), ['solarize', 'affine', 'perspective', 'rotation'])

type(transforms_final(img))