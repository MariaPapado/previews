import random
import numpy as np
from PIL import Image
# from scipy import misc
import torch
import torchvision
import torchvision.transforms.functional as F
import time
from PIL import ImageEnhance
import sys
#imagenet mean std 
#def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):   #imgnet

#def normalize_img(img, mean=[105.451, 91.725, 130.431], std=[77.773, 58.877, 73.883]):  #with stretch
def normalize_img(img, mean=[113.645, 109.630, 142.233], std=[67.524, 59.366, 67.692]):  #without stretch

    """Normalize image by subtracting mean and dividing by std."""
    img_array = np.asarray(img, dtype=np.uint8)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img

def random_fliplr(img1, mask):
    if random.random() > 0.5:
        img1 = np.fliplr(img1)
        mask = np.fliplr(mask)

    return img1, mask


def random_flipud(img1, mask):
    if random.random() > 0.5:
        img1 = np.flipud(img1)
        mask = np.flipud(mask)

    return img1, mask


def random_rot(img1, img2, mask):
    k = random.randrange(3) + 1

    img1 = np.rot90(img1, k).copy()
    img2 = np.rot90(img2, k).copy()
    mask = np.rot90(mask, k).copy()

    return img1, img2, mask


