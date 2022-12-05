import numpy as np
import requests
from matplotlib import pyplot as plt
from PIL import Image

if __name__=='__main__':
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
    img = img/255.0

    def show_image(image, title=''):
        # image is [H, W, 3]
        assert image.shape[2] == 3
        plt.imshow(np.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).astype(int))
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.show()
        return


    plt.rcParams['figure.figsize'] = [5, 5]
    img_tensor = img
    show_image(img_tensor)