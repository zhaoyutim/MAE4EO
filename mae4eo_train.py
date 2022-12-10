import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import numpy as np
import requests


import platform

import yaml
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint
from PIL import Image
import wandb as wandb
import tensorflow_addons as tfa
import tensorflow as tf
from wandb.integration.keras import WandbCallback

from cosine_warmup import WarmUpCosine
from image_patch_division import ImagePatchDivision
from model import MaskedAutoencoder, create_encoder, create_decoder
from patch_encoder import PatchEncoder
from utils import ImagenetLoader
import tensorflow.python.keras.backend as K

# from tensorflow.keras import mixed_precision
#
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

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
def visualize_output(model, test_image, patch_encoder, epoch):
    test_image = model.random_resize_and_crop_test(test_image)
    test_image = model.input_norm(test_image)
    test_patches = model.patch_layer(test_image)

    (   test_unmasked_embeddings,
        test_masked_tokens,
        test_unmasked_positions,
        test_mask_indices,
        test_unmask_indices,
        test_mask_restore
    ) = model.patch_encoder(test_image)

    # all masked_token
    cls_token = test_unmasked_embeddings[:, :1, :]
    encoder_inputs = tf.concat([test_unmasked_embeddings[:, 1:, :], test_masked_tokens], axis=1)
    encoder_inputs = tf.gather(encoder_inputs, test_mask_restore, axis=1, batch_dims=1)
    encoder_inputs = tf.concat([cls_token, encoder_inputs], axis=1)

    encoder_outputs = model.encoder(encoder_inputs)
    test_decoder_inputs = model.encoder_decoder_projection(encoder_outputs)


    test_decoder_outputs = model.decoder(test_decoder_inputs)

    # Show a maksed patch image.
    test_masked_patch, idx = patch_encoder.generate_masked_image(test_patches, test_unmask_indices)
    print(f"\nIdx chosen: {idx}")
    import numpy as np
    image_mean = np.array([0.485, 0.456, 0.406])
    image_std = np.array([0.229, 0.224, 0.225])
    original_image = test_image[idx]*image_std+image_mean
    patch_layer = ImagePatchDivision(patch_size=PATCH_SIZE)
    masked_image = patch_layer.reconstruct_from_patch(test_masked_patch)

    reconstructed_image = patch_layer.reconstruct_from_patch(test_decoder_outputs[idx,1:,:])

    masked_image = np.where(masked_image == 0, 0, masked_image.numpy()*image_std+image_mean)
    reconstructed_image = reconstructed_image.numpy()*image_std+image_mean
    reconstructed_image_with_known_patch = np.where(masked_image == 0, reconstructed_image, masked_image)

    # reconstructed_image_with_known_patch = np.where
    # reconstructed_image = test_decoder_outputs[idx]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))
    ax[0][0].imshow(np.clip(original_image,0,1))
    ax[0][0].set_title(f"Original:")

    ax[0][1].imshow(masked_image)
    ax[0][1].set_title(f"Masked:")

    ax[1][0].imshow(np.clip(reconstructed_image,0,1))
    ax[1][0].set_title(f"Resonstructed:")

    ax[1][1].imshow(np.clip(reconstructed_image_with_known_patch,0,1))
    ax[1][1].set_title(f"Resonstructed with Original Patches:")
    # plt.savefig('checkpoints_image/output_at_epoch_'+str(epoch)+'.png')
    plt.show()
    plt.close()

with open("dataset_config.yml", "r", encoding="utf8") as f:
    dataset_config = yaml.load(f, Loader=yaml.FullLoader)

with open("model_config.yml", "r", encoding="utf8") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)



if __name__ == '__main__':
    # Setting seeds for reproducibility.
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
    percent='100'

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

    def mse(y_true, y_pred):
        squared_error = K.square(y_true - y_pred)
        masked_mse = K.sum(K.mean(K.mean(squared_error, axis=2), axis=1))
        return masked_mse

    if training == 'train':
        wandb_config(epochs=EPOCHS, model_name='mae4eo', batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    imagenet = ImagenetLoader(dataset=dataset, percent=percent)
    train_gen, val_gen, test_gen = imagenet.dataset_generator(dataset=dataset, batch_size=BATCH_SIZE, augment=True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        patch_layer = ImagePatchDivision(patch_size=PATCH_SIZE)
        patch_encoder = PatchEncoder(patch_size=PATCH_SIZE, projection_dim=ENC_PROJECTION_DIM,
                                     mask_proportion=MASK_PROPORTION, patch_division=patch_layer, decoder_projection_dim =DEC_PROJECTION_DIM, num_patches=NUM_PATCHES)
        encoder = create_encoder(num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS,
                                 encoder_projection_dim=ENC_PROJECTION_DIM,
                                 encoder_transformer_units=ENC_TRANSFORMER_UNITS, layer_norm_eps=LAYER_NORM_EPS)
        decoder = create_decoder(num_pathces=NUM_PATCHES, num_heads=DEC_NUM_HEADS, image_size=IMAGE_SIZE,
                                 patch_size=PATCH_SIZE, num_layers=DEC_LAYERS,
                                 encoder_projection_dim=ENC_PROJECTION_DIM,
                                 decoder_projection_dim=DEC_PROJECTION_DIM,
                                 decoder_transformer_units=DEC_TRANSFORMER_UNITS, layer_norm_eps=LAYER_NORM_EPS)
        mae_model = MaskedAutoencoder(
            image_shape=(IMAGE_SIZE, IMAGE_SIZE),
            patch_layer=patch_layer,
            patch_encoder=patch_encoder,
            encoder=encoder,
            decoder=decoder
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
        optimizer = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, beta_1=0.9, beta_2=0.95)
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
    model_path='mae4eo/mae4eowf_batchsize' + str(BATCH_SIZE) + '_lr_' + str(LEARNING_RATE) + '_mask_' + str(mask_proportion)+'_model_'+model+'_dataset_'+dataset + percent+'_percent'
    checkpoint_path = 'mae4eo_checkpoints/mae4eowf_batchsize' + str(BATCH_SIZE) + '_lr_' + str(LEARNING_RATE) + '_mask_' + str(mask_proportion)+'_model_'+model+'_dataset_'+dataset + percent+'_percent'
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss", mode="min", save_best_only=True, verbose=1)


    class TrainMonitor(tf.keras.callbacks.Callback):
        def __init__(self, epoch_interval=None):
            self.epoch_interval = epoch_interval

        def on_epoch_end(self, epoch, logs=None):
            if self.epoch_interval and epoch % self.epoch_interval == 0:
                visualize_output(self.model, next(iter(test_gen)), self.model.patch_encoder, epoch)


    train_callbacks = [TrainMonitor(epoch_interval=5)]
    if training == 'train':
        history = mae_model.fit(
            train_gen,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=int((imagenet.len_val/BATCH_SIZE)),
            steps_per_epoch=int((imagenet.len_train / BATCH_SIZE)),
            callbacks=[WandbCallback(save_model=False), checkpoint, train_callbacks]
        )
        mae_model.save(model_path)
    else:
        mae_model.load_weights(checkpoint_path)
    # load an image
    img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'  # fox, from ILSVRC2012_val_00046145
    # img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    # normalize by ImageNet mean and std
    # img = img - imagenet_mean
    # img = img / imagenet_std

    def show_image(image, title=''):
        # image is [H, W, 3]
        assert image.shape[2] == 3
        plt.imshow(np.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).astype(int))
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.show()
        return

    plt.rcParams['figure.figsize'] = [5, 5]
    img_tensor = tf.convert_to_tensor(img[np.newaxis, :,:,:])
    show_image(img)
    # loss, mae = mae_model.evaluate(test_gen, steps=imagenet.len_test/BATCH_SIZE)
    # print(f"Loss: {loss:.2f}")
    # print(f"MAE: {mae:.2f}")
    # visualize_output(mae_model, next(iter(test_gen)), patch_encoder)
    visualize_output(mae_model, img_tensor, patch_encoder, EPOCHS)
