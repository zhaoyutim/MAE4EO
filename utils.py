import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers
import tensorflow_datasets as tfds
AUTOTUNE = tf.data.AUTOTUNE

class ImagenetLoader:
    def __init__(self,dataset,
                 data_dir = '~/tensorflow_datasets/downloads/ilsvrc2012',
                 write_dir = '~/tensorflow_datasets/imagenet2012'):
        # Construct a tf.data.Dataset
        if dataset=='imagenet':
            download_config = tfds.download.DownloadConfig(
                extract_dir=os.path.join(write_dir, 'extracted'),
                manual_dir=data_dir
            )
            download_and_prepare_kwargs = {
                'download_dir': os.path.join(write_dir, 'downloaded'),
                'download_config': download_config,
            }
            self.x_train, self.x_val, self.x_test = tfds.load('imagenet2012',
                                       data_dir=os.path.join(write_dir, 'data'),
                                       split=['train[:20%]', 'validation', 'test'],
                                        shuffle_files=True,
                                        download=True,
                                        as_supervised=True,
                                        with_info=False,
                                       download_and_prepare_kwargs=download_and_prepare_kwargs)
        else:
            (self.x_train, y_train), (self.x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            (self.x_train, y_train), (self.x_val, y_val) = (
                (self.x_train[:40000], y_train[:40000]),
                (self.x_train[40000:], y_train[40000:]),
            )
            self.x_train = tf.data.Dataset.from_tensor_slices(self.x_train)
            self.x_val = tf.data.Dataset.from_tensor_slices(self.x_val)
            self.x_test = tf.data.Dataset.from_tensor_slices(self.x_test)
        self.len_train = len(self.x_train)
        self.len_val = len(self.x_val)
        self.len_test = len(self.x_test)

    def random_crop(self, image):
        cropped_image = tf.image.random_crop(
          image, size=[224, 224, 3])
        return cropped_image

    def rescaling(self, image):
        image = tf.cast(image, tf.float32)
        image = tf.divide(image, 255.0)
        return image

    def resizing(self, image):
        image = tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image

    def random_crop_cifar(self, image):
        cropped_image = tf.image.random_crop(
          image, size=[48, 48, 3])
        return cropped_image

    def resizing_cifar_train(self, image):
        image = tf.image.resize(image, [52, 52], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image

    def resizing_cifar_test(self, image):
        image = tf.image.resize(image, [48, 48], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image

    def training_augmentation(self, image, label):
        image = self.rescaling(image)
        image = self.resizing(image)
        image = self.random_crop(image)
        image = tf.image.random_flip_left_right(image)
        return image

    def test_augmentation(self, image, label):
        image = self.rescaling(image)
        image = self.resizing(image)
        image = self.random_crop(image)
        return image

    def training_augmentation_cifar(self, image):
        image = self.rescaling(image)
        image = self.resizing_cifar_train(image)
        image = self.random_crop_cifar(image)
        image = tf.image.random_flip_left_right(image)
        return image

    def test_augmentation_cifar(self, image):
        image = self.rescaling(image)
        image = self.resizing_cifar_test(image)
        return image

    def dataset_generator(self, dataset, batch_size=32, augment=False):
        self.BATCH_SIZE = batch_size

        train_dataset = self.x_train.apply(tf.data.experimental.ignore_errors())
        val_dataset = self.x_val.apply(tf.data.experimental.ignore_errors())
        test_dataset = self.x_test.apply(tf.data.experimental.ignore_errors())

        train_dataset = train_dataset.shuffle(batch_size * 10).repeat()
        val_dataset = val_dataset.shuffle(batch_size * 10).repeat()
        test_dataset = test_dataset.shuffle(batch_size)
        if dataset=='imagenet':
            train_dataset = train_dataset.map(self.training_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            val_dataset = val_dataset.map(self.training_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_dataset = test_dataset.map(self.test_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            train_dataset = train_dataset.map(self.training_augmentation_cifar, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            val_dataset = val_dataset.map(self.training_augmentation_cifar, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_dataset = test_dataset.map(self.test_augmentation_cifar, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset
# def load_data(dataset='cifar10'):
#         data_dir = '~/tensorflow_datasets/downloads/ilsvrc2012'
#         write_dir = '~/tensorflow_datasets/imagenet2012'
#
#         # Construct a tf.data.Dataset
#         download_config = tfds.download.DownloadConfig(
#             extract_dir=os.path.join(write_dir, 'extracted'),
#             manual_dir=data_dir
#         )
#         download_and_prepare_kwargs = {
#             'download_dir': os.path.join(write_dir, 'downloaded'),
#             'download_config': download_config,
#         }
#         x_train, x_val = tfds.load('imagenet2012',
#                        data_dir=os.path.join(write_dir, 'data'),
#                        split=['train', 'validation'],
#                        shuffle_files=False,
#                        download=False,
#                        as_supervised=False,
#                        download_and_prepare_kwargs=download_and_prepare_kwargs)
#     print(f"Training samples: {len(x_train)}")
#     print(f"Validation samples: {len(x_val)}")
#     assert isinstance(x_train, tf.data.Dataset)
#     assert isinstance(x_val, tf.data.Dataset)
#     return x_train, x_val, len(x_train)
# def imagenet_generator(dataset, batch_size):
#
#     images = np.zeros((batch_size, 224, 224, 3))
#     while True:
#         count = 0
#         for sample in tfds.as_numpy(dataset):
#             image = sample["image"]
#             images[count % batch_size] = training_augmentation(image)
#             count += 1
#             if (count % batch_size == 0):
#                 yield images
if __name__ == '__main__':
    imagenet = ImagenetLoader('cifar')
    image_gen, val_gen, test_gen = imagenet.dataset_generator(dataset='cifar',batch_size=256, augment=True)
    image_batch = next(iter(val_gen)).numpy()
    print(image_batch)
    plt.imshow(image_batch[0,:,:,:])
    plt.show()