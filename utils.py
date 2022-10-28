import os

import tensorflow as tf
from tensorflow.keras import layers

def get_train_augmentation_model(input_shape, image_size):
    model = tf.keras.Sequential(
        [
            layers.Rescaling(1 / 255.0),
            layers.Resizing(input_shape[1] + 20, input_shape[1] + 20),
            layers.RandomCrop(image_size, image_size),
            layers.RandomFlip("horizontal"),
        ],
        name="train_data_augmentation",
    )
    return model


def get_test_augmentation_model(image_size):
    model = tf.keras.Sequential(
        [layers.Rescaling(1 / 255.0), layers.Resizing(image_size, image_size),],
        name="test_data_augmentation",
    )
    return model

def load_data(batch_size, buffer_size, dataset='cifar10', mode='train'):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if mode=='train':
        (x_train, y_train), (x_val, y_val) = (
            (x_train[:40000], y_train[:40000]),
            (x_train[40000:], y_train[40000:]),
        )
    else:
        (x_train, y_train), (x_val, y_val) = (
            (x_train[:400], y_train[:400]),
            (x_train[400:800], y_train[400:800]),
        )
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Testing samples: {len(x_test)}")

    train_ds = tf.data.Dataset.from_tensor_slices(x_train)
    train_ds = train_ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(x_val)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(x_test)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds, len(x_train), len(x_test)