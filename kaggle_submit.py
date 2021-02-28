from keras.applications.inception_v3 import InceptionV3
from keras_preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras import optimizers, losses
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import pickle

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

base_model = InceptionV3(weights=None, include_top=False, input_shape=(300, 300, 3))
base_model.load_weights('../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.05),
    Dense(1024, activation='relu'),
    Dense(6, activation='softmax')
])

opt = optimizers.nadam(lr=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

batch_size = 200
epochs = 500
train_generator = gen["train"]
valid_generator = gen["valid"]

steps_per_epoch = train_generator.n // batch_size
validation_steps = valid_generator.n // batch_size

filepath = "model_{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]
history = model.fit_generator(generator=train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                              validation_data=valid_generator, validation_steps=validation_steps,
                              callbacks=callbacks_list)
with open('trainHistoryDict.txt', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)