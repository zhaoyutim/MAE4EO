import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf

from image_patch_division import ImagePatchDivision
from utils import get_train_augmentation_model, load_data


class PatchEncoder(layers.Layer):
    def __init__(
        self,
        patch_size,
        projection_dim,
        mask_proportion,
        downstream=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream

        # This is a trainable mask token initialized randomly from a normal
        # distribution.
        self.mask_token = tf.Variable(
            tf.random.normal([1, patch_size * patch_size * 3]), trainable=True
        )

    def build(self, input_shape):
        (_, self.num_patches, self.patch_area) = input_shape

        # Create the projection layer for the patches.
        self.projection = layers.Dense(units=self.projection_dim)

        # Create the positional embedding layer.
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )

        # Number of patches that will be masked.
        self.num_mask = int(self.mask_proportion * self.num_patches)

    def call(self, patches):
        # Get the positional embeddings.
        batch_size = tf.shape(patches)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1, 1]
        )  # (B, num_patches, projection_dim)

        # Embed the patches.
        patch_embeddings = (
            self.projection(patches) + pos_embeddings
        )  # (B, num_patches, projection_dim)

        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            # The encoder input is the unmasked patch embeddings. Here we gather
            # all the patches that should be unmasked.
            unmasked_embeddings = tf.gather(
                patch_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)

            # Get the unmasked and masked position embeddings. We will need them
            # for the decoder.
            unmasked_positions = tf.gather(
                pos_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)
            masked_positions = tf.gather(
                pos_embeddings, mask_indices, axis=1, batch_dims=1
            )  # (B, mask_numbers, projection_dim)

            # Repeat the mask token number of mask times.
            # Mask tokens replace the masks of the image.
            mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            mask_tokens = tf.repeat(
                mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
            )

            # Get the masked embeddings for the tokens.
            masked_embeddings = self.projection(mask_tokens) + masked_positions
            return (
                unmasked_embeddings,  # Input to the encoder.
                masked_embeddings,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                mask_indices,  # The indices that were masked.
                unmask_indices,  # The indices that were unmaksed.
            )

    def get_random_indices(self, batch_size):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

    def generate_masked_image(self, patches, unmask_indices):
        # Choose a random patch and it corresponding unmask index.
        idx = np.random.choice(patches.shape[0])
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
    PATCH_SIZE = 6
    train_ds, val_ds, test_ds, len_train, len_test = load_data(batch_size=256, buffer_size=1024)
    image_batch = next(iter(train_ds))
    augmentation_model = get_train_augmentation_model(image_size=48, input_shape=(32, 32, 3))
    augmented_images = augmentation_model(image_batch)
    patch_layer = ImagePatchDivision(patch_size=PATCH_SIZE)
    patches = patch_layer(images=augmented_images)
    patch_encoder = PatchEncoder(patch_size=PATCH_SIZE, projection_dim=128, mask_proportion=0.75)
    (unmasked_embeddings,masked_embeddings,unmasked_positions,mask_indices,unmask_indices,) = patch_encoder(patches=patches)

    # Show a maksed patch image.
    new_patch, random_index = patch_encoder.generate_masked_image(patches, unmask_indices)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    img = patch_layer.reconstruct_from_patch(new_patch)
    plt.imshow(tf.keras.utils.array_to_img(img))
    plt.axis("off")
    plt.title("Masked")
    plt.subplot(1, 2, 2)
    img = augmented_images[random_index]
    plt.imshow(tf.keras.utils.array_to_img(img))
    plt.axis("off")
    plt.title("Original")
    plt.show()