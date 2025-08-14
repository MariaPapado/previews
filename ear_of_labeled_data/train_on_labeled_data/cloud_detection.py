import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import imutils

class CloudDetection(Dataset):
    def __init__(self, dataset_path, data_list, type='train'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        
        self.type = type

    def stretch_16bit(self, image, percentile_min=2, percentile_max=98):
        tmp = []
        for i in range(image.shape[2]):
            image_band = image[:, :, i].astype(np.float32)
            min_val = np.nanmin(image_band[np.nonzero(image_band)])
            max_val = np.nanmax(image_band[np.nonzero(image_band)])

            if max_val>min_val:
                image_band = (image_band - min_val) / (max_val - min_val)
            else:
                image_band = image_band / 65535.

            image_band_valid = image_band[np.logical_and(image_band > 0, image_band < 1.0)]
            perc_2 = np.nanpercentile(image_band_valid, percentile_min)
            perc_98 = np.nanpercentile(image_band_valid, percentile_max)

            if perc_98>perc_2:
                band = (image_band - perc_2) / (perc_98 - perc_2)
            else:
                band = image_band.copy()

            band[band < 0] = 0.0
            band[band > 1] = 1.0
            tmp.append(band)
        return np.stack(tmp, 2)


    def normalize_img(self, img, mean=[0.360, 0.360, 0.269], std=[0.293, 0.263, 0.264]):  #without stretch

        """Normalize image by subtracting mean and dividing by std."""
        #img_array = np.asarray(img, dtype=np.uint8)
        img_array = img
        normalized_img = np.empty_like(img_array)

        for i in range(3):  # Loop over color channels
            normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
        
        return normalized_img


    def normalize_single_img(self, img):  #without stretch

        means = img.mean(axis=(0, 1))
        stds = img.std(axis=(0, 1))
        if 0 in stds:
            print('zero stds!!!!!', stds)
            stds = [0.5, 0.5, 0.5]
        normalized_img = (img - means) / stds

        return normalized_img


    def __transforms(self, aug, img1, mask):
        if aug:
            img1, mask = imutils.random_fliplr(img1, mask)
            img1, mask = imutils.random_flipud(img1, mask)
        #    img1, img2, mask = imutils.random_rot(img1, img2, mask)

        img1 = self.normalize_single_img(img1)  # imagenet normalization

        img1 = np.transpose(img1, (2, 0, 1))

        return img1, mask

    def __getitem__(self, index):
            
        img1 = Image.open(os.path.join(self.dataset_path, 'images', self.data_list[index]))
        mask = Image.open(os.path.join(self.dataset_path, 'labels', self.data_list[index]))


        img1, mask = np.array(img1)/255., np.array(mask)/255.
#        print('aaaaa', img1.shape, mask.shape)

        if 'train' in self.type:
            img1, mask = self.__transforms(True, img1, mask)
        else:
            img1, mask = self.__transforms(False, img1, mask)

        #mask = np.expand_dims(mask, 0)

        data_idx = self.data_list[index]
        return np.ascontiguousarray(img1), np.ascontiguousarray(mask)
#        return np.array(img2, dtype=float), np.array(mask, dtype=float), label, data_idx

    def __len__(self):
        return len(self.data_list)









