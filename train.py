from keras.callbacks import ModelCheckpoint

import data_gen
import my_model

batch_size = 32
epochs = 10


def train_and_valid(model, data_generator):
    train_generator = data_generator["train"]
    valid_generator = data_generator["valid"]

    steps_per_epoch = train_generator.n // batch_size
    validation_steps = valid_generator.n // batch_size

    filepath = "model_{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    history = model.fit_generator(generator=train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  validation_data=valid_generator, validation_steps=validation_steps,
                                  callbacks=callbacks_list)


if __name__ == '__main__':

    pass
