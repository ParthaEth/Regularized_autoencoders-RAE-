import os
from imageio import imwrite
import numpy as np


def save_set_of_images(path, images):
    if not os.path.exists(path):
        os.mkdir(path)

    images = (np.clip(images, 0, 1) * 255).astype('uint8')

    for i, img in enumerate(images):
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        imwrite(os.path.join(path, '%08d.png' % i), img)