import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf

from image_patch_division import ImagePatchDivision
from pos_embed import get_2d_sincos_pos_embed
from utils import ImagenetLoader


class PatchEncoder(layers.Layer):
    def __init__(
        self,
        patch_size,
        num_patches,
        projection_dim,
        decoder_projection_dim,
        mask_proportion,
        patch_division,
        downstream=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.decoder_projection_dim = decoder_projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream
        self.patch_division = patch_division
        self.cls_token = tf.Variable(tf.random.normal([1, 1, self.projection_dim], stddev=0.02), trainable=True)
        # This is a trainable mask token initialized randomly from a normal
        # distribution.
        self.mask_token = tf.Variable(tf.random.normal([1, self.decoder_projection_dim], stddev=0.02), trainable=True)
        self.projection_cnn = layers.Conv2D(filters=self.projection_dim, kernel_size=patch_size, strides=patch_size)
        pos_embed = get_2d_sincos_pos_embed(projection_dim, int(num_patches ** .5), cls_token=True)
        self.position_embedding = tf.convert_to_tensor(pos_embed, dtype=tf.float32)


    def build(self, input_shape):

        self.num_patches = (input_shape[1]//self.patch_size) ** 2
        # Create the projection layer for the patches.
        # self.projection = layers.Dense(units=self.projection_dim)

        # Create the positional embedding layer.
        # self.position_embedding = layers.Embedding(
        #     input_dim=self.num_patches+1, output_dim=self.projection_dim
        # )

        # Number of patches that will be masked.
        self.num_mask = int(self.mask_proportion * self.num_patches)

    def call(self, images):
        # Get the positional embeddings.

        batch_size = tf.shape(images)[0]

        # positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        pos_embeddings = tf.tile(
            self.position_embedding[tf.newaxis], [batch_size, 1, 1]
        )  # (B, num_patches, projection_dim)

        # Embed the patches.
        # patch_embeddings = (self.projection(patches) + pos_embeddings)  # (B, num_patches, projection_dim)
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        patch_embeddings = self.projection_cnn(images)
        # patches = self.patch_layer(images=images)
        patch_embeddings = tf.reshape(patch_embeddings, [-1, self.num_patches, self.projection_dim])
        patch_embeddings = tf.concat([cls_tokens, patch_embeddings], 1)
        patch_embeddings = (patch_embeddings + pos_embeddings)  # (B, num_patches, projection_dim)
        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices, mask_restore = self.get_random_indices(batch_size)
            # The encoder input is the unmasked patch embeddings. Here we gather
            # all the patches that should be unmasked.
            unmasked_embeddings = tf.gather(
                patch_embeddings[:,1:,:], unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)
            unmasked_embeddings = tf.concat([patch_embeddings[:,:1,:], unmasked_embeddings], axis=1)
            # Get the unmasked and masked position embeddings. We will need them
            # for the decoder.
            unmasked_positions = tf.gather(
                pos_embeddings[:, 1:, :], unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)
            masked_positions = tf.gather(
                pos_embeddings[:, 1:, :], mask_indices, axis=1, batch_dims=1
            )  # (B, mask_numbers, projection_dim)
            unmasked_positions = tf.concat([pos_embeddings[:,:1,:], unmasked_positions], axis=1)
            # Repeat the mask token number of mask times.
            # Mask tokens replace the masks of the image.
            mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            mask_tokens = tf.repeat(
                mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
            )

            # Get the masked embeddings for the tokens.
            # masked_embeddings = mask_tokens + masked_positions
            return (
                unmasked_embeddings,  # Input to the encoder.
                mask_tokens,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                mask_indices,  # The indices that were masked.
                unmask_indices,  # The indices that were unmaksed.
                mask_restore,
            )

    def get_random_indices(self, batch_size):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_restore = tf.argsort(rand_indices, axis=-1)
        unmask_indices = rand_indices[:, : self.num_patches-self.num_mask]
        mask_indices = rand_indices[:, self.num_patches-self.num_mask :]
        return mask_indices, unmask_indices, mask_restore

    def generate_masked_image(self, patches, unmask_indices):
        # Choose a random patch and it corresponding unmask index.
        # idx = np.random.choice(patches.shape[0])
        idx = 0
        patch = patches[idx]
        unmask_index = unmask_indices[idx]

        # Build a numpy array of same shape as patch.
        new_patch = np.zeros_like(patch)

        # Iterate of the new_patch and plug the unmasked patches.
        count = 0
        for i in range(unmask_index.shape[0]):
            new_patch[unmask_index[i]] = patch[unmask_index[i]]
        return new_patch, idx

if __name__=='__main__':
    PATCH_SIZE = 16
    imagenet = ImagenetLoader(dataset='imagenet')
    image_gen, val_gen, test_gen = imagenet.dataset_generator(dataset='imagenet', batch_size=256, augment=True)
    image_batch = next(iter(image_gen)).numpy()
    patch_layer = ImagePatchDivision(patch_size=PATCH_SIZE)
    patches = patch_layer(images=image_batch)
    patch_encoder = PatchEncoder(patch_size=PATCH_SIZE, num_patches=196, projection_dim=768, decoder_projection_dim=512, mask_proportion=0.75, patch_division=patch_layer)
    (unmasked_embeddings,masked_embeddings,unmasked_positions,mask_indices,unmask_indices,mask_restore) = patch_encoder(images=image_batch)

    # Show a maksed patch image.
    new_patch, random_index = patch_encoder.generate_masked_image(patches, unmask_indices)
    # patch_restore = tf.gather(tf.concat([unmasked_embeddings, masked_embeddings], 1), mask_restore, axis=1,
    #                           batch_dims=1)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    img = patch_layer.reconstruct_from_patch(new_patch)
    plt.imshow(tf.keras.utils.array_to_img(img))
    plt.axis("off")
    plt.title("Masked")
    plt.subplot(1, 2, 2)
    img = image_batch[random_index]
    plt.imshow(tf.keras.utils.array_to_img(img))
    plt.axis("off")
    plt.title("Original")
    plt.show()

    # plt.subplot(1,3,3)
    # img = patch_layer.reconstruct_from_patch(patch_restore[0, :, :])
    # plt.imshow(tf.keras.utils.array_to_img(img))
    # plt.axis("off")
    # plt.title("Masked Emb")
    # plt.show()


