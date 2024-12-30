import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random

TRAIN_DIR = 'train'
TEST_DIR = 'test'

train_dataset = image_dataset_from_directory(TRAIN_DIR,
                                             shuffle=True,
                                             batch_size=32,
                                             # Reducimos el valor 
                                             # de la imagen
                                             image_size=(150,150))

test_dataset = image_dataset_from_directory(TEST_DIR,
                                            shuffle=True,
                                            batch_size=32,
                                            image_size=(150,150))

class_names = train_dataset.class_names
print(class_names)


def plot_imgs(images, labels):
    plt.figure(figsize=(10,10))
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3,3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])

plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu')
])