#import sys
#sys.path.append('/notebooks/GeoSeg/')
#import segmentation_models_pytorch as smp 
import os
import tools
import torch
import numpy as np
import rasterio as rio
import warnings
import cv2
import cloud_detection
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import shutil
#import torchnet as tnt
import argparse
from unet_model import *
from unet_parts import *
from PIL import Image
print('ok')


def normalize_single_img(img):  #without stretch

    means = img.mean(axis=(0, 1))
    stds = img.std(axis=(0, 1))
    normalized_img = (img - means) / stds

    return normalized_img


device = 'cuda' if torch.cuda.is_available() else 'cpu'


net = UNet(n_channels=3, n_classes=2, bilinear=True)
net.load_state_dict(torch.load('./saved_models/net_16.pt'))
net.to(device)
#print(net)
net=net.eval()

test_ids = os.listdir('/home/maria/previews_clouds/DATASET_previews_TEST/picture/')

for i, id in enumerate(tqdm(test_ids)):
    print(id)
    imgs = Image.open('/home/maria/previews_clouds/DATASET_previews_TEST/picture/{}'.format(id))

    imgs = np.array(imgs)/255.


    imgs = normalize_single_img(imgs)
    imgs = torch.from_numpy(imgs).float().permute(2,0,1).unsqueeze(0).cuda()

#################################################################################################################################


#################################################################################################################################


#    imgs, labels = imgs.float().to(device), labels.long().to(device)

    preds= net(imgs)
    preds = torch.argmax(preds,1).data.cpu().numpy()
#    print(preds.shape)
    cv2.imwrite('./whole_preview_tests/pred_{}.png'.format(id), preds[0]*256)


