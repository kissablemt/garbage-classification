from keras_preprocessing.image import ImageDataGenerator

data_dir = 'Garbage classification'


def get_data_generator(img_shape):
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
            target_size=img_shape,
            subset='training',
        ),

        "valid": ImageDataGenerator(
            rescale=1 / 255,
            validation_split=0.1,
        ).flow_from_directory(
            directory=data_dir,
            target_size=img_shape,
            subset='validation',
        ),
    }

    return gen


if __name__ == '__main__':
    data_gen = get_data_generator((300, 300))
    # print(data_gen["train"].class_indices)
    print(len(data_gen["train"].classes))
    pass