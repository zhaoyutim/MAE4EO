from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf

from image_patch_division import ImagePatchDivision
from patch_encoder import PatchEncoder
from utils import load_data, get_train_augmentation_model, get_test_augmentation_model


class MaskedAutoencoder(tf.keras.Model):
    def __init__(
        self,
        train_augmentation_model,
        test_augmentation_model,
        patch_layer,
        patch_encoder,
        encoder,
        decoder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_augmentation_model = train_augmentation_model
        self.test_augmentation_model = test_augmentation_model
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder

    def calculate_loss(self, images, test=False):
        # Augment the input images.
        if test:
            augmented_images = self.test_augmentation_model(images)
        else:
            augmented_images = self.train_augmentation_model(images)

        # Patch the augmented images.
        patches = self.patch_layer(augmented_images)

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches)

        # Pass the unmaksed patche to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        # Decode the inputs.
        decoder_outputs = self.decoder(decoder_inputs)
        decoder_patches = self.patch_layer(decoder_outputs)

        loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
        loss_output = tf.gather(decoder_patches, mask_indices, axis=1, batch_dims=1)

        # Compute the total loss.
        total_loss = self.compiled_loss(loss_patch, loss_output)

        return total_loss, loss_patch, loss_output

    def train_step(self, images):
        with tf.GradientTape() as tape:
            total_loss, loss_patch, loss_output = self.calculate_loss(images)

        # Apply gradients.
        train_vars = [
            self.train_augmentation_model.trainable_variables,
            self.patch_layer.trainable_variables,
            self.patch_encoder.trainable_variables,
            self.encoder.trainable_variables,
            self.decoder.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        # Report progress.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        total_loss, loss_patch, loss_output = self.calculate_loss(images, test=True)

        # Update the trackers.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):
        # Augment the input images.
        if training:
            augmented_images = self.train_augmentation_model(inputs)
        else:
            augmented_images = self.test_augmentation_model(inputs)

        # Patch the augmented images.
        patches = self.patch_layer(augmented_images)

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches)

        # Pass the unmaksed patche to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        # Decode the inputs.
        decoder_outputs = self.decoder(decoder_inputs)

        return decoder_outputs

def mlp(x, dropout_rate, hidden_units):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_encoder(num_heads, num_layers, encoder_projection_dim, encoder_transformer_units, layer_norm_eps):
    inputs = layers.Input((None, encoder_projection_dim))
    x = inputs

    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=layer_norm_eps)(x)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=encoder_projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=layer_norm_eps)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=encoder_transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x = layers.Add()([x3, x2])

    outputs = layers.LayerNormalization(epsilon=layer_norm_eps)(x)
    return tf.keras.Model(inputs, outputs, name="mae_encoder")


def create_decoder(num_pathces, num_layers, num_heads, image_size, encoder_projection_dim, decoder_projection_dim,
                   decoder_transformer_units, layer_norm_eps):

    inputs = layers.Input((num_pathces, encoder_projection_dim))
    x = layers.Dense(decoder_projection_dim)(inputs)
    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=layer_norm_eps)(x)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=decoder_projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=layer_norm_eps)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=decoder_transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=layer_norm_eps)(x)
    x = layers.Flatten()(x)
    pre_final = layers.Dense(units=image_size * image_size * 3, activation="sigmoid")(x)
    outputs = layers.Reshape((image_size, image_size, 3))(pre_final)

    return tf.keras.Model(inputs, outputs, name="mae_decoder")

if __name__=='__main__':
    BUFFER_SIZE = 1024
    BATCH_SIZE = 256
    INPUT_SHAPE = (32, 32, 3)
    NUM_CLASSES = 10

    # OPTIMIZER
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 1e-46

    # PRETRAINING
    EPOCHS = 1

    # AUGMENTATION
    IMAGE_SIZE = 48  # We will resize input images to this size.
    PATCH_SIZE = 6  # Size of the patches to be extracted from the input images.
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
    MASK_PROPORTION = 0.75  # We have found 75% masking to give us the best results.

    # ENCODER and DECODER
    LAYER_NORM_EPS = 1e-6
    ENC_PROJECTION_DIM = 128
    DEC_PROJECTION_DIM = 64
    ENC_NUM_HEADS = 4
    ENC_LAYERS = 6
    DEC_NUM_HEADS = 4
    DEC_LAYERS = (
        2  # The decoder is lightweight but should be reasonably deep for reconstruction.
    )
    ENC_TRANSFORMER_UNITS = [
        ENC_PROJECTION_DIM * 2,
        ENC_PROJECTION_DIM,
    ]  # Size of the transformer layers.
    DEC_TRANSFORMER_UNITS = [
        DEC_PROJECTION_DIM * 2,
        DEC_PROJECTION_DIM,
    ]
    train_augmentation_model = get_train_augmentation_model(image_size=IMAGE_SIZE, input_shape=INPUT_SHAPE)
    test_augmentation_model = get_test_augmentation_model(image_size=IMAGE_SIZE)
    train_augmentation_model.build(input_shape=(None, 32, 32, 3))
    test_augmentation_model.build(input_shape=(None, 32, 32, 3))
    patch_layer = ImagePatchDivision(patch_size=PATCH_SIZE)
    patch_encoder = PatchEncoder(patch_size=PATCH_SIZE, projection_dim=ENC_PROJECTION_DIM, mask_proportion=MASK_PROPORTION)
    encoder = create_encoder(num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS, encoder_projection_dim=ENC_PROJECTION_DIM,
                             encoder_transformer_units=ENC_TRANSFORMER_UNITS, layer_norm_eps=LAYER_NORM_EPS)
    decoder = create_decoder(num_pathces=NUM_PATCHES, image_size=IMAGE_SIZE, num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS, encoder_projection_dim=ENC_PROJECTION_DIM,
                             decoder_projection_dim=DEC_PROJECTION_DIM, decoder_transformer_units=DEC_TRANSFORMER_UNITS, layer_norm_eps=LAYER_NORM_EPS)
    mae_model = MaskedAutoencoder(
        train_augmentation_model=train_augmentation_model,
        test_augmentation_model=test_augmentation_model,
        patch_layer=patch_layer,
        patch_encoder=patch_encoder,
        encoder=encoder,
        decoder=decoder
    )
    mae_model.build(input_shape=(None, 32, 32, 3))
    mae_model.summary()
