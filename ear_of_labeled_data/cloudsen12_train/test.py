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
import MANet
import myUNF
from unet_model import *
from unet_parts import *
print('ok')


device = 'cuda' if torch.cuda.is_available() else 'cpu'


#net = myUNF.UNetFormer(num_classes=4)
#net=MANet.MANet(3,4)
net = UNet(n_channels=3, n_classes=4, bilinear=True)
net.load_state_dict(torch.load('./saved_models/net_16.pt'))
net.to(device)
#print(net)
net=net.eval()



val_batch_size=1


valset = cloud_detection.CloudDetection('./', 'test', 975)
print('1')

valloader = DataLoader(valset, batch_size=val_batch_size, shuffle=False,
                                    pin_memory=True, drop_last=False)



cm_total = np.zeros((4,4))
for i, batch in enumerate(tqdm(valloader)):
#    Ximg, y = batch
    #print(Ximg.shape, y.shape)
   # print('uni', np.unique(y.data.cpu().numpy()))

    imgs, labels = batch
#################################################################################################################################

    img_save = imgs[0].permute(1,2,0).data.cpu().numpy()
    lbl_save = labels[0].squeeze().data.cpu().numpy()

    cv2.imwrite('./check/images/img_{}.png'.format(i), img_save[:,:,[2,1,0]]*255)
    cv2.imwrite('./check/masks/lbl_{}.png'.format(i), lbl_save.astype(np.uint8)*80)

#################################################################################################################################


    imgs, labels = imgs.float().to(device), labels.long().to(device)

    preds= net(imgs)
    preds = torch.argmax(preds,1).data.cpu().numpy()
    cv2.imwrite('./check/labels/pred_{}.png'.format(i), preds[0]*80)


    label_conf, pred_conf = labels.data.cpu().numpy().flatten(), preds.flatten()
    cm = confusion_matrix(label_conf, pred_conf, labels=[0, 1,2,4])
    cm_total += cm
    


    #scheduler.step(np.mean(val_losses))    
    test_acc=(np.trace(cm_total)/float(np.ndarray.sum(cm_total))) *100
    #prec, rec, f1 = tools.metrics(cm_total)
    #print ('Precision: {}\nRecall: {}\nF1: {}'.format(prec, rec, f1)) 
    

    #tools.write_results(ff, save_folder, epoch, train_acc, test_acc, np.mean(train_losses), np.mean(val_losses), cm_total, prec, rec, f1, optimizer.param_groups[0]['lr'])

    #save model in every epoch
print(cm_total)
