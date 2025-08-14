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
import albumentations as A
import albumentations.pytorch
import torchvision.transforms.functional as F


class CloudDetection(Dataset):

    def __init__(self, root, mode, nb):
        super(CloudDetection, self).__init__()
        self.root = root
        self.mode = mode
        self.nb = nb

        B2X = np.expand_dims(np.memmap(os.path.join(self.root, self.mode, 'L2A_B2.dat'), dtype='int16', mode='r', shape=(self.nb, 512, 512)), 3)
        B3X = np.expand_dims(np.memmap(os.path.join(self.root, self.mode, 'L2A_B3.dat'), dtype='int16', mode='r', shape=(self.nb, 512, 512)), 3)
        B4X = np.expand_dims(np.memmap(os.path.join(self.root, self.mode, 'L2A_B4.dat'), dtype='int16', mode='r', shape=(self.nb, 512, 512)), 3)

#        B8X = np.expand_dims(np.memmap(os.path.join(self.root, self.mode, 'L2A_B8.dat'), dtype='int16', mode='r', shape=(self.nb, 512, 512)), 3)

#        self.y = np.expand_dims(np.memmap(os.path.join(self.root, self.mode, 'LABEL_manual_hq.dat'), dtype='int8', mode='r', shape=(self.nb, 512, 512)), 3)
        self.y = np.memmap(os.path.join(self.root, self.mode, 'LABEL_manual_hq.dat'), dtype='int8', mode='r', shape=(self.nb, 512, 512))
        print('AAAAAAAAAAAAAA')

        self.X = np.concatenate((B4X, B3X, B2X), 3)
        self.X = self.X
        self.y = self.y


    def normalize_img(self, img, mean=[0.360, 0.360, 0.269], std=[0.293, 0.263, 0.264]):  #without stretch


        """Normalize image by subtracting mean and dividing by std."""
        #img_array = np.asarray(img, dtype=np.uint8)
        img_array = img
        normalized_img = np.empty_like(img_array)

        for i in range(3):  # Loop over color channels
            normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
        
        return normalized_img


    def normalize_single_img(self, img):  #without stretch

        """Normalize image by subtracting mean and dividing by std."""
        #img_array = np.asarray(img, dtype=np.uint8)


        #normalized_img = np.empty_like(img)
        # Normalize each channel

        means = img.mean(axis=(0, 1))
        stds = img.std(axis=(0, 1))
        if 0 in stds:
            print('zero stds!!!!!', stds)
            stds = [0.5, 0.5, 0.5]
        normalized_img = (img - means) / stds

#        for i in range(3):  # Loop over color channels
#            normalized_img[..., i] = (img[..., i] - means[i]) / stds[i]


        return normalized_img

    def stretch_16bittttt(self, band, lower_percent=2, higher_percent=98):
        a = 0
        b = 65535
        real_values = band.flatten()
        real_values = real_values[real_values > 0]
        c = np.percentile(real_values, lower_percent)
        d = np.percentile(real_values, higher_percent)
        t = a + (band - c) * (b - a) / float(d - c)
        t[t<a] = a
        t[t>b] = b

        t = (t / 256).astype(np.uint8)
#        t = t.astype(np.uint8) #/255.
        return t/255.  #/65535.

#        return t/65535.


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


    def __getitem__(self, index):


        Ximg = np.array(self.X[index].copy())
        yimg = np.array(self.y[index].copy()) #.squeeze()
#        img_save = Ximg
#        lbl_save = yimg

#        idx2 = np.where(yimg==2)
#        idx3 = np.where(yimg==3)
#        yimg[idx2]=0
#        yimg[idx3]=0


        #print('uniiiiiii', np.unique(Ximg))
        #cv2.imwrite('./check/labels/lbl_{}.png'.format(index), yimg*20)


        X_stretched = self.stretch_16bit(Ximg)
        X_stretched = X_stretched*255.
        X_stretched = X_stretched.astype(np.uint8)
        X_stretched = X_stretched/255.
        X_stretched = self.normalize_single_img(X_stretched)
#        cv2.imwrite('./check/images/img_{}.png'.format(index), X_stretched*255)
#        cv2.imwrite('./check/labels/lbl_{}.png'.format(index), lbl_save*256)

        
#        X_stretched, yimg = X_stretched.astype(np.float32), yimg.astype(np.float32)   
#        cv2.imwrite('./checks/img_{}.png'.format(index), X_stretched[:,:,[2,1,0]]*255)

        X_stretched = self.normalize_single_img(X_stretched)

#        X_stretched = cv2.resize(X_stretched, (256,256), interpolation=cv2.INTER_NEAREST)
#        yimg = cv2.resize(yimg, (256,256), interpolation=cv2.INTER_NEAREST)
        X_stretched = np.transpose(X_stretched, (2,0,1))

        return X_stretched, yimg

    def __len__(self):
        return len(self.y)
    
