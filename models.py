from keras import optimizers
from keras.applications import InceptionV3, ResNet50
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential


def model1():
    # 0.79
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', input_shape=(300, 300, 3), activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(6, activation='softmax'),
    ])
    opt = optimizers.adam()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    return model


def model2():
    # v13-0.80
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', input_shape=(300, 300, 3), activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(6, activation='softmax'),
    ])
    opt = optimizers.adam()
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])


def model3():
    # v17-0.804
    base_model = InceptionV3(weights=None, include_top=False, input_shape=(300, 300, 3))
    base_model.load_weights('../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dense(6, activation='softmax')
    ])
    opt = optimizers.adam(lr=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    return model


def model4():
    # v16-0.836
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
    return model


def model5():
    # v19-0.88
    base_model = InceptionV3(weights=None, include_top=False, input_shape=(300, 300, 3))
    base_model.load_weights('../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.02),
        Dense(1024, activation='relu'),
        Dense(2, activation='softmax')
    ])
    opt = optimizers.nadam(lr=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    return model


def model6():
    base_model = InceptionV3(weights=None, include_top=False, input_shape=(300, 300, 3))
    base_model.load_weights('../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.25),
        Dense(1024, activation='relu'),
        Dense(6, activation='softmax')
    ])
    opt = optimizers.nadam(lr=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    return model


def model7():
    base_model = ResNet50(weights=None, include_top=False, input_shape=(300, 300, 3))
    base_model.load_weights('../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(6, activation='softmax')
    ])
    opt = optimizers.nadam(lr=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    return model


def model8():
    base_model = ResNet50(weights=None, include_top=False, input_shape=(300, 300, 3))
    base_model.load_weights('../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.50),
        Dense(6, activation='softmax')
    ])
    opt = optimizers.nadam(lr=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    return model
