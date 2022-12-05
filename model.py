import os
import yaml
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf

from image_patch_division import ImagePatchDivision
from patch_encoder import PatchEncoder
from pos_embed import get_2d_sincos_pos_embed
from utils import ImagenetLoader

class RandomResizedCrop(tf.keras.layers.Layer):
    # taken from
    # https://keras.io/examples/vision/nnclr/#random-resized-crops
    def __init__(self, scale=(0.2, 1.0), ratio=(3.0/4.0, 4.0/3.0), crop_shape=(48,48)):
        super(RandomResizedCrop, self).__init__()
        self.scale = scale
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
        self.crop_shape = crop_shape

    def call(self, images):
        batch_size = tf.shape(images)[0]

        random_scales = tf.random.uniform(
            (batch_size,),
            self.scale[0],
            self.scale[1]
        )
        random_ratios = tf.exp(tf.random.uniform(
            (batch_size,),
            self.log_ratio[0],
            self.log_ratio[1]
        ))

        new_heights = tf.clip_by_value(
            tf.sqrt(random_scales / random_ratios),
            0,
            1,
        )
        new_widths = tf.clip_by_value(
            tf.sqrt(random_scales * random_ratios),
            0,
            1,
        )
        height_offsets = tf.random.uniform(
            (batch_size,),
            0,
            1 - new_heights,
        )
        width_offsets = tf.random.uniform(
            (batch_size,),
            0,
            1 - new_widths,
        )

        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )
        images = tf.image.crop_and_resize(
            image=images,
            boxes=bounding_boxes,
            box_indices=tf.range(batch_size),
            crop_size=self.crop_shape,
            method='bilinear',
        )

        images = tf.image.resize(images=images, size=self.crop_shape, method='bicubic')
        return images

class MaskedAutoencoder(tf.keras.Model):
    def __init__(
        self,
        image_shape,
        patch_layer,
        patch_encoder,
        encoder,
        decoder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_decoder_projection = layers.Dense(units=decoder.input_shape[2])
        self.input_norm = layers.Normalization(axis=-1, mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
        self.random_resize_and_crop = RandomResizedCrop(scale=(0.2, 1.0), crop_shape=image_shape)
        self.random_resize_and_crop_test = RandomResizedCrop(scale=(1.0, 1.0), crop_shape=image_shape)
        self.random_flip = layers.RandomFlip(mode='horizontal')

    def calculate_loss(self, images, test=False):
        # Patch the augmented images.
        images = self.random_resize_and_crop(images)
        images = self.random_flip(images)
        images = self.input_norm(images)
        patches = self.patch_layer(images)

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_tokens,
            unmasked_positions,
            mask_indices,
            unmask_indices,
            mask_restore
        ) = self.patch_encoder(images)

        # Pass the unmaksed patche to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings)

        # Create the decoder inputs.
        # encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = self.encoder_decoder_projection(encoder_outputs)
        decoder_inputs = tf.concat([decoder_inputs, masked_tokens], axis=1)

        cls_token = decoder_inputs[:, :1, :]
        decoder_inputs = tf.gather(decoder_inputs[:, 1:, :], mask_restore, axis=1, batch_dims=1)
        decoder_inputs = tf.concat([cls_token, decoder_inputs], axis=1)

        # Decode the inputs.
        decoder_outputs = self.decoder(decoder_inputs)
        # decoder_patches = self.patch_layer(decoder_outputs)
        # loss_output = tf.gather(decoder_patches, mask_indices, axis=1, batch_dims=1)

        loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
        loss_output = tf.gather(decoder_outputs[:,1:,:], mask_indices, axis=1, batch_dims=1)
        # loss_output = tf.gather(decoder_patches, mask_indices, axis=1, batch_dims=1)

        # Compute the total loss.
        total_loss = self.compiled_loss(loss_patch, loss_output)

        return total_loss, loss_patch, loss_output

    def train_step(self, images):
        with tf.GradientTape() as tape:
            total_loss, loss_patch, loss_output = self.calculate_loss(images)

        # Apply gradients.
        train_vars = [
            # self.train_augmentation_model.trainable_variables,
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
        images = self.random_resize_and_crop_test(inputs)
        images = self.input_norm(images)
        patches = self.patch_layer(images)
        # Encode the patches.
        (
            unmasked_embeddings,
            masked_token,
            unmasked_positions,
            mask_indices,
            unmask_indices,
            mask_restore,
        ) = self.patch_encoder(inputs)

        # Pass the unmaksed patche to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings)

        # Create the decoder inputs.
        # encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = self.encoder_decoder_projection(encoder_outputs)
        decoder_inputs = tf.concat([decoder_inputs, masked_token], axis=1)

        cls_token = decoder_inputs[:, :1, :]
        decoder_inputs = tf.gather(decoder_inputs[:, 1:, :], mask_restore, axis=1, batch_dims=1)
        decoder_inputs = tf.concat([cls_token, decoder_inputs], axis=1)

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
            num_heads=num_heads, key_dim=encoder_projection_dim//num_heads, dropout=0.0
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=layer_norm_eps)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=encoder_transformer_units, dropout_rate=0.0)

        # Skip connection 2.
        x = layers.Add()([x3, x2])

    outputs = layers.LayerNormalization(epsilon=layer_norm_eps)(x)
    return tf.keras.Model(inputs, outputs, name="mae_encoder")


def create_decoder(num_pathces, num_layers, num_heads, image_size, patch_size, encoder_projection_dim, decoder_projection_dim,
                   decoder_transformer_units, layer_norm_eps):

    pos_embed = get_2d_sincos_pos_embed(decoder_projection_dim, int(num_pathces ** .5), cls_token=True)
    dec_position_embedding = tf.convert_to_tensor(pos_embed, dtype=tf.float32)

    inputs = layers.Input((num_pathces+1, decoder_projection_dim))
    # x = layers.Dense(decoder_projection_dim)(inputs)
    x = inputs + dec_position_embedding
    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=layer_norm_eps)(x)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=decoder_projection_dim//num_heads, dropout=0.0
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=layer_norm_eps)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=decoder_transformer_units, dropout_rate=0.0)

        # Skip connection 2.
        x = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=layer_norm_eps)(x)
    outputs = layers.Dense(units=patch_size * patch_size * 3, activation="linear")(x)
    # x = layers.Flatten()(x)
    # pre_final = layers.Dense(units=image_size * image_size * 3, activation="linear")(x)
    # outputs = layers.Reshape((image_size, image_size, 3))(pre_final)

    return tf.keras.Model(inputs, outputs, name="mae_decoder")

if __name__=='__main__':

    with open("dataset_config.yml", "r", encoding="utf8") as f:
        dataset_config = yaml.load(f, Loader=yaml.FullLoader)

    with open("model_config.yml", "r", encoding="utf8") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    dataset='imagenet'
    BUFFER_SIZE, \
    BATCH_SIZE, \
    IMG_SHAPE, \
    NUM_CLASSES, \
    N_CHANNELS, \
    IMAGE_SIZE, \
    PATCH_SIZE,\
    LEARNING_RATE = dataset_config.get(dataset).values()
    INPUT_SHAPE = (None, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS)
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
    MASK_PROPORTION = 0.75

    # OPTIMIZER
    WEIGHT_DECAY = 1e-4

    # ENCODER and DECODER
    LAYER_NORM_EPS, \
    ENC_PROJECTION_DIM, \
    ENC_NUM_HEADS, \
    ENC_LAYERS,\
    ENC_MLP_RATIO, \
    DEC_PROJECTION_DIM, \
    DEC_NUM_HEADS, \
    DEC_LAYERS,\
    DEC_MLP_RATIO = model_config.get('vit_base').values()

    ENC_TRANSFORMER_UNITS = [
        ENC_PROJECTION_DIM * ENC_MLP_RATIO,
        ENC_PROJECTION_DIM
    ]
    DEC_TRANSFORMER_UNITS = [
        DEC_PROJECTION_DIM * DEC_MLP_RATIO,
        DEC_PROJECTION_DIM
    ]
    patch_layer = ImagePatchDivision(patch_size=PATCH_SIZE)
    patch_encoder = PatchEncoder(patch_size=PATCH_SIZE, projection_dim=ENC_PROJECTION_DIM, mask_proportion=MASK_PROPORTION, decoder_projection_dim=512, patch_division=patch_layer, num_patches=NUM_PATCHES)
    encoder = create_encoder(num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS, encoder_projection_dim=ENC_PROJECTION_DIM,
                             encoder_transformer_units=ENC_TRANSFORMER_UNITS, layer_norm_eps=LAYER_NORM_EPS)

    # encoder.compute_output_shape(input_shape=INPUT_SHAPE)

    decoder = create_decoder(num_pathces=NUM_PATCHES, num_heads=DEC_NUM_HEADS, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_layers=DEC_LAYERS, encoder_projection_dim=ENC_PROJECTION_DIM,
                             decoder_projection_dim=DEC_PROJECTION_DIM, decoder_transformer_units=DEC_TRANSFORMER_UNITS, layer_norm_eps=LAYER_NORM_EPS)

    mae_model = MaskedAutoencoder(
        image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        patch_layer=patch_layer,
        patch_encoder=patch_encoder,
        encoder=encoder,
        decoder=decoder
    )

    mae_model.compute_output_shape(input_shape=INPUT_SHAPE)
    mae_model.summary()
    mae_model.encoder.summary()
    mae_model.decoder.summary()

