import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dc1.image_dataset import ImageDataset
import torch
from pathlib import Path

# 'raw data', this is how the data looks like when you just read the files.
X_test = np.load('data/X_test.npy')   # Consists of the pictures we want to test on
X_train = np.load('data/X_train.npy')  # Consists of the pictures we want to train on
Y_test = np.load('data/Y_test.npy')   # Consists of the classes of the corresponding X_test pictures
Y_train = np.load('data/Y_train.npy')  # Consists of the classes of the corresponding X_train pictures

# this is how we load the data for most projects. The data here is an ImageDataset class.
train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

# classes
classes = {0:'Atelectasis', 1:'Effusion', 2:'Infiltration', 3: 'No Finding', 4:'Nodule', 5:'Pneumothorax'}

picture = X_train[0]
print(picture.shape)  # returns (1, 128, 128) (image with a single channel (grayscale) and a resolution of 128x128 pixels)
print(type(picture))
print(picture)  # prints the representation of a numpy array
print(len(X_test))  # X_test data set consists of 8420 pictures
print(len(X_train))  # X_test data set consists of 16841 pictures

print(len(X_test) == len(Y_test))  # returns true
print(len(X_train) == len(Y_train))  # returns true

print(Y_test[0])  # The first class in the Y_test is 3

# Plotting the stacked bar chart that is on Canvas
unique_classes, counts_Y_train = np.unique(Y_train, return_counts=True)
unique_classes, counts_Y_test = np.unique(Y_test, return_counts=True)

plt.bar(unique_classes, counts_Y_train)
plt.bar(unique_classes, counts_Y_test, bottom=counts_Y_train)
plt.xlabel('Class')
plt.xticks([0,1,2,3,4,5], ['Atelectasis','Effusion','Infiltration','No Finding','Nodule','Pneumothorax'])
plt.ylabel('No. of images')
plt.title('Distribution of image labels')
plt.legend(['Training', 'Test'])
plt.show()

# plotting the first picture of the train data
plt.imshow(picture.reshape(128, 128), cmap='gray')
plt.title(f'Image of class: {Y_train[0]}')
plt.show()

# Making a DataFrame of the unique classes and their counts
unique, counts = np.unique(Y_train, return_counts=True)
print(pd.DataFrame({'Class': unique, 'Counts': counts}))

# An ImageDataset class has two methods:
# .targets method, which shows the targets of the images
# .imgs method, which shows the 3 dimensional arrays of all images
print(train_dataset.targets[0])
print(train_dataset.imgs[0])

for i in range(len(train_dataset.imgs)):
    if train_dataset.imgs[i].shape != (1, 128, 128):
        print(f'The {i}th image is not complete.')



print('----------------------------------')
print(train_dataset.imgs[0][0][10][7])
print('----------------------------------')

for i in range(40):
    if train_dataset.imgs[i][0][10][7] > 150:
        plt.imshow(train_dataset.imgs[i].reshape(128, 128), cmap='gray')
        plt.title(f'Image of class: {train_dataset.targets[i]}')
        plt.show()
        print(f'the {i}th image of class {train_dataset.targets[i]}')

unique, counts = np.unique(train_dataset.targets, return_counts=True)
print(dict(zip(unique, counts)))
# In oder to get balanced data, we need (all classes have 6103 images then):
# 3582 more images of class 0
# 3785 more images of class 1
# 3139 more images of class 2
# 0 more images of class 3
# 4470 more images of class 4
# 4801 more images of class 5
print(unique)
