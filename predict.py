from keras.applications.inception_v3 import InceptionV3
from keras_preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras import optimizers, losses
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

data_dir = '../input/garbage-classification/Garbage classification/Garbage classification'
# data_dir = '../Garbage classification'

gen = {
    "train": ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1. / 255,
        validation_split=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=30,
    ).flow_from_directory(
        directory=data_dir,
        target_size=(300, 300),
        subset='training',
    ),

    "valid": ImageDataGenerator(
        rescale=1 / 255,
        validation_split=0.1,
    ).flow_from_directory(
        directory=data_dir,
        target_size=(300, 300),
        subset='validation',
    ),
}

model = load_model('../input/best-model/the_best_model.h5')

labels = gen["train"].class_indices
labels = dict((v, k) for k, v in labels.items())

test_x, test_y = gen["valid"].__getitem__(1)
preds = model.predict(test_x)

plt.figure(figsize=(16, 16))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.title('pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])
