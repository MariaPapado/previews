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



def postprocess(image):

    # Find contours
    contours_first, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for con in contours_first:
        area = cv2.contourArea(con)
        if area>400:
            contours.append(con)

    output = cv2.drawContours(np.zeros((image.shape[0], image.shape[1],3)), contours, -1, (255,255,255), thickness=cv2.FILLED)

    # Smooth the mask
    blurred_mask = cv2.GaussianBlur(output, (25, 25), 0)

    # Threshold back to binary
    _, smoothed_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

    return smoothed_mask

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
    preds = np.transpose(preds, (1,2,0))


    rgb_preds = postprocess(preds.astype(np.uint8))

    cv2.imwrite('./whole_preview_tests/pred_{}.png'.format(id), rgb_preds)


