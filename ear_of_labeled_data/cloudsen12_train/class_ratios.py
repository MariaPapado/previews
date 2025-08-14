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


    def stretch_16bit(self, band, lower_percent=2, higher_percent=98):
        a = 0
        b = 65535
        real_values = band.flatten()
        real_values = real_values[real_values > 0]
        c = np.percentile(real_values, lower_percent)
        d = np.percentile(real_values, higher_percent)
        t = a + (band - c) * (b - a) / float(d - c)
        t[t<a] = a
        t[t>b] = b
#        t = t/256.
#        t = t.astype(np.uint8) #/255.
        return t/65535.
        


    def __getitem__(self, index):


        Ximg = np.array(self.X[index].copy())
        yimg = np.array(self.y[index].copy()).squeeze()

        

        X_stretched = self.stretch_16bit(Ximg)
        #X_stretched = Ximg/65535.        
#        X_stretched, yimg = X_stretched.astype(np.float32), yimg.astype(np.float32)   
#        X_stretched = cv2.resize(X_stretched, (256,256), interpolation=cv2.INTER_NEAREST)
#        yimg = cv2.resize(yimg, (256,256), interpolation=cv2.INTER_NEAREST)
        X_stretched = np.transpose(X_stretched, (2,0,1))

        return X_stretched, yimg

    def __len__(self):
        return len(self.y)
    



trainset = CloudDetection('./', 'train', 8490)
print('0')
trainloader = DataLoader(trainset, batch_size=1, shuffle=True,
                                      pin_memory=False, drop_last=False)

class_sums = [0]*4

for i, batch in enumerate(tqdm(trainloader)):

    Ximg, y = batch
    mask = y[0].data.numpy()


    for c in range(0, len(class_sums)):
        idx = np.where(mask==c)
        count = len(idx[0])
        class_sums[c] = class_sums[c] + count

total_samples = 8490*512*512

for i in range(0, len(class_sums)):
    print('class {} '.format(i), (class_sums[i]/total_samples)*100)


# Count occurrences of each class
class_counts = np.array(class_sums)

# Calculate class frequencies
class_freqs = class_counts / total_samples
print("Class frequencies:", class_freqs)

# Calculate alpha for each class as the inverse of class frequency
alpha = 1.0 / class_freqs
print("Alpha values:", alpha)


# Normalize alpha to sum to 1
alpha /= alpha.sum()
print("Normalized alpha values:", alpha)
