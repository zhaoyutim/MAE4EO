import argparse
import os
import platform

import yaml
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint

import wandb as wandb
import tensorflow_addons as tfa
import tensorflow as tf
from wandb.integration.keras import WandbCallback

from cosine_warmup import WarmUpCosine
from image_patch_division import ImagePatchDivision
from model import MaskedAutoencoder, create_encoder, create_decoder
from patch_encoder import PatchEncoder
from utils import ImagenetLoader

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
def visualize_output(model, test_image):
    test_patches = model.patch_layer(test_image)

    (   test_unmasked_embeddings,
        test_masked_embeddings,
        test_unmasked_positions,
        test_mask_indices,
        test_unmask_indices,
        test_mask_restore
    ) = model.patch_encoder(test_patches)

    test_encoder_outputs = model.encoder(test_unmasked_embeddings)
    test_encoder_outputs = test_encoder_outputs + test_unmasked_positions

    test_decoder_inputs = tf.concat([test_encoder_outputs, test_masked_embeddings], axis=1)

    cls_token = test_decoder_inputs[:, :1, :]
    test_decoder_inputs = tf.gather(test_decoder_inputs[:, 1:, :], test_mask_restore, axis=1, batch_dims=1)
    test_decoder_inputs = tf.concat([cls_token, test_decoder_inputs], axis=1)


    test_decoder_outputs = model.decoder(test_decoder_inputs)
    patch_encoder = PatchEncoder(patch_size=PATCH_SIZE, projection_dim=ENC_PROJECTION_DIM, mask_proportion=MASK_PROPORTION)

    # Show a maksed patch image.
    test_masked_patch, idx = patch_encoder.generate_masked_image(test_patches, test_unmask_indices)
    print(f"\nIdx chosen: {idx}")
    original_image = test_image[idx]
    patch_layer = ImagePatchDivision(patch_size=PATCH_SIZE)
    masked_image = patch_layer.reconstruct_from_patch(test_masked_patch)

    reconstructed_image = patch_layer.reconstruct_from_patch(test_decoder_outputs[idx,1:,:])
    # reconstructed_image = test_decoder_outputs[idx]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax[0].imshow(original_image)
    ax[0].set_title(f"Original:")

    ax[1].imshow(masked_image)
    ax[1].set_title(f"Masked:")

    ax[2].imshow(reconstructed_image)
    ax[2].set_title(f"Resonstructed:")

    plt.show()
    plt.close()

with open("dataset_config.yml", "r", encoding="utf8") as f:
    dataset_config = yaml.load(f, Loader=yaml.FullLoader)

with open("model_config.yml", "r", encoding="utf8") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    # Setting seeds for reproducibility.
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    SEED = 42
    tf.random.set_seed(SEED)
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-mp', type=float, help='masked proportion')
    parser.add_argument('-t', type=str, help='training or testing')
    parser.add_argument('-epoch', type=int, help='training ephoches')
    parser.add_argument('-d', type=str, help='dataset')
    parser.add_argument('-m', type=str, help='model')
    args = parser.parse_args()
    mask_proportion = args.mp
    training = args.t
    EPOCHS = args.epoch
    dataset= args.d
    model = args.m

    # DATA and AUGMENTATION
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
    MASK_PROPORTION = mask_proportion

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
    DEC_MLP_RATIO = model_config.get(model).values()

    ENC_TRANSFORMER_UNITS = [
        ENC_PROJECTION_DIM * ENC_MLP_RATIO,
        ENC_PROJECTION_DIM
    ]
    DEC_TRANSFORMER_UNITS = [
        DEC_PROJECTION_DIM * DEC_MLP_RATIO,
        DEC_PROJECTION_DIM
    ]
    if training == 'train':
        wandb_config(epochs=EPOCHS, model_name='mae4eo', batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    imagenet = ImagenetLoader(dataset=dataset)
    train_gen, val_gen, test_gen = imagenet.dataset_generator(dataset=dataset, batch_size=BATCH_SIZE, augment=True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        patch_layer = ImagePatchDivision(patch_size=PATCH_SIZE)
        patch_encoder = PatchEncoder(patch_size=PATCH_SIZE, projection_dim=ENC_PROJECTION_DIM, mask_proportion=MASK_PROPORTION)
        encoder = create_encoder(num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS, encoder_projection_dim=ENC_PROJECTION_DIM,
                                 encoder_transformer_units=ENC_TRANSFORMER_UNITS, layer_norm_eps=LAYER_NORM_EPS)
        decoder = create_decoder(num_pathces=NUM_PATCHES, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_heads=ENC_NUM_HEADS, num_layers=DEC_LAYERS, encoder_projection_dim=ENC_PROJECTION_DIM,
                                 decoder_projection_dim=DEC_PROJECTION_DIM, decoder_transformer_units=DEC_TRANSFORMER_UNITS, layer_norm_eps=LAYER_NORM_EPS)
        mae_model = MaskedAutoencoder(
            patch_layer=patch_layer,
            patch_encoder=patch_encoder,
            encoder=encoder,
            decoder=decoder,
        )
        mae_model.compute_output_shape(input_shape=INPUT_SHAPE)
        total_steps = int((imagenet.len_train / BATCH_SIZE) * EPOCHS)
        warmup_epoch_percentage = 0.15
        warmup_steps = int(total_steps * warmup_epoch_percentage)
        scheduled_lrs = WarmUpCosine(
            learning_rate_base=LEARNING_RATE,
            total_steps=total_steps,
            warmup_learning_rate=0.0,
            warmup_steps=warmup_steps,
        )
        optimizer = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mae_model.encoder.summary()
    mae_model.decoder.summary()
    mae_model.summary()
    # Compile and pretrain the model.
    mae_model.compile(
        optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"]
    )
    # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs',
    #                                                  histogram_freq=1,
    #                                                  profile_batch='20, 40')
    model_path='mae4eo/mae4eo_batchsize' + str(BATCH_SIZE) + '_lr_' + str(LEARNING_RATE) + '_mask_' + str(mask_proportion)+'_model_'+model
    if training == 'train':
        history = mae_model.fit(
            train_gen,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=int((imagenet.len_val/BATCH_SIZE)),
            steps_per_epoch=int((imagenet.len_train / BATCH_SIZE)),
            callbacks=[WandbCallback(save_model=False)]
        )
        mae_model.save(model_path)
    else:
        mae_model.load_weights(model_path)

    loss, mae = mae_model.evaluate(test_gen, steps=imagenet.len_test/BATCH_SIZE)
    print(f"Loss: {loss:.2f}")
    print(f"MAE: {mae:.2f}")
    visualize_output(mae_model, next(iter(test_gen)))