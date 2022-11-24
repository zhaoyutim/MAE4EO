import os

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf

from utils import ImagenetLoader


class ImagePatchDivision(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.resize = layers.Reshape((-1, patch_size * patch_size * 3))

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = self.resize(patches)
        return patches

    def show_patched_image(self, images, patches):
        idx = np.random.choice(patches.shape[0])
        print(f"Index selected: {idx}.")

        plt.figure(figsize=(4, 4))
        plt.imshow(tf.keras.utils.array_to_img(images[idx]))
        plt.axis("off")
        plt.show()

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[idx]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (self.patch_size, self.patch_size, 3))
            plt.imshow(tf.keras.utils.img_to_array(patch_img))
            plt.axis("off")
        plt.show()
        return idx

    def reconstruct_from_patch(self, patch):
        num_patches = patch.shape[0]
        n = int(np.sqrt(num_patches))
        patch = tf.reshape(patch, (num_patches, self.patch_size, self.patch_size, 3))
        rows = tf.split(patch, n, axis=0)
        rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
        reconstructed = tf.concat(rows, axis=0)
        return reconstructed

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    PATCH_SIZE = 6
    imagenet = ImagenetLoader(dataset='cifar')
    image_gen, val_gen, test_gen = imagenet.dataset_generator(dataset='cifar', batch_size=256, augment=True)
    image_batch = next(iter(image_gen)).numpy()
    patch_layer = ImagePatchDivision(patch_size=PATCH_SIZE)
    patches = patch_layer(images=image_batch)
    random_index = patch_layer.show_patched_image(images=image_batch, patches=patches)
    image = patch_layer.reconstruct_from_patch(patches[random_index])
    plt.imshow(image)
    plt.axis("off")
    plt.show()