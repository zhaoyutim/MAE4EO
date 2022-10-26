import argparse

from tensorflow.python.keras.callbacks import ModelCheckpoint

import wandb as wandb
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf
from wandb.integration.keras import WandbCallback

from cosine_warmup import WarmUpCosine
from image_patch_division import ImagePatchDivision
from model import MaskedAutoencoder
from patch_encoder import PatchEncoder
from utils import load_data, get_train_augmentation_model, get_test_augmentation_model


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

def wandb_config(epochs, model_name, batch_size, learning_rate):
    wandb.login()
    wandb.init(project='mae4eo', entity="zhaoyutim")
    wandb.run.name = 'model_name' + str(model_name) + 'batchsize_'+str(batch_size)+'learning_rate_'+str(learning_rate)
    wandb.config = {
      "learning_rate": learning_rate,
      "epochs": epochs,
      "batch_size": batch_size,
      "model_name":model_name,
    }

if __name__ == '__main__':
    # Setting seeds for reproducibility.
    SEED = 42
    tf.random.set_seed(SEED)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-lr', type=float, help='learning rate')
    parser.add_argument('-t', type=str, help='training or testing')
    args = parser.parse_args()
    batch_size = args.b
    learning_rate = args.lr
    training = args.t
    # DATA
    BUFFER_SIZE = 1024
    BATCH_SIZE = batch_size
    INPUT_SHAPE = (32, 32, 3)
    NUM_CLASSES = 10

    # OPTIMIZER
    LEARNING_RATE = learning_rate
    WEIGHT_DECAY = 1e-4

    # PRETRAINING
    EPOCHS = 100

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
    if training == 'train':
        wandb_config(epochs=EPOCHS, model_name='mae4eo', batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

    train_ds, val_ds, test_ds, len_train, len_test = load_data(BATCH_SIZE, BUFFER_SIZE)
    train_augmentation_model = get_train_augmentation_model(image_size=IMAGE_SIZE, input_shape=INPUT_SHAPE)
    test_augmentation_model = get_test_augmentation_model(image_size=IMAGE_SIZE)
    test_augmentation_model.build(input_shape=(None, 32, 32, 3))
    patch_layer = ImagePatchDivision(patch_size=PATCH_SIZE)
    patch_encoder = PatchEncoder(patch_size=PATCH_SIZE, projection_dim=ENC_PROJECTION_DIM, mask_proportion=MASK_PROPORTION)
    encoder = create_encoder(num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS, encoder_projection_dim=ENC_PROJECTION_DIM,
                             encoder_transformer_units=ENC_TRANSFORMER_UNITS, layer_norm_eps=LAYER_NORM_EPS)
    decoder = create_decoder(num_pathces=NUM_PATCHES, image_size=IMAGE_SIZE, num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS, encoder_projection_dim=ENC_PROJECTION_DIM,
                             decoder_projection_dim=DEC_PROJECTION_DIM, decoder_transformer_units=DEC_TRANSFORMER_UNITS, layer_norm_eps=LAYER_NORM_EPS)
    if training != 'train':
        encoder=tf.keras.models.load_model('mae4eo/encoder_batchsize'+str(batch_size)+'_lr_'+str(learning_rate))
        decoder=tf.keras.models.load_model('mae4eo/decoder_batchsize' + str(batch_size) + '_lr_' + str(learning_rate))
    mae_model = MaskedAutoencoder(
        train_augmentation_model=train_augmentation_model,
        test_augmentation_model=test_augmentation_model,
        patch_layer=patch_layer,
        patch_encoder=patch_encoder,
        encoder=encoder,
        decoder=decoder,
    )
    total_steps = int((len_train / BATCH_SIZE) * EPOCHS)
    warmup_epoch_percentage = 0.15
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=LEARNING_RATE,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )
    optimizer = tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)
    # Compile and pretrain the model.
    mae_model.compile(
        optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"]
    )
    if training == 'train':
        history = mae_model.fit(
            train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[WandbCallback(save_model=False)],
        )
        encoder.save('mae4eo/encoder_batchsize'+str(batch_size)+'_lr_'+str(learning_rate))
        decoder.save('mae4eo/decoder_batchsize' + str(batch_size) + '_lr_' + str(learning_rate))

    # Measure its performance.
    loss, mae = mae_model.evaluate(test_ds)
    print(f"Loss: {loss:.2f}")
    print(f"MAE: {mae:.2f}")