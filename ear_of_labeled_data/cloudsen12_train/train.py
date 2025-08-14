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
#net.load_state_dict(torch.load('./saved_models/net_23.pt'))
net.to(device)
#print(net)
w_tensor=torch.FloatTensor(4)
w_tensor[0]= 0.07
w_tensor[1]= 0.14
w_tensor[2]= 0.38
w_tensor[3]= 0.41

w_tensor = w_tensor.to(device)

criterion = torch.nn.CrossEntropyLoss(w_tensor).to(device)
#criterion = torch.nn.CrossEntropyLoss().to(device)


base_lr = 0.0001
base_wd = 0.01

#layerwise_params = {"backbone.*": dict(lr=base_lr, weight_decay=base_wd)}
#net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
#base_optimizer = torch.optim.AdamW(net_params, lr=base_lr, weight_decay=base_wd)
#optimizer = Lookahead(base_optimizer)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)


optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
#optimizer = torch.optim.AdamW(net.parameters(), lr=base_lr, weight_decay=base_wd)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=2, verbose=True)

batch_size = 8
val_batch_size=8

epochs = 30

print('loaders')

trainset = cloud_detection.CloudDetection('./', 'train', 8490)
print('0')
valset = cloud_detection.CloudDetection('./', 'val', 535)
print('1')
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                      pin_memory=False, drop_last=False)
print('2')
valloader = DataLoader(valset, batch_size=val_batch_size, shuffle=False,
                                    pin_memory=True, drop_last=False)
print('3')

print('calculating alpha..')
#alpha = get_alpha(trainloader)
# alpha-0 (no-cloud)=1216263864, alpha-1 (thick-cloud)=596674914, alpha-2 (thin-cloud)=215404610, alpha-3 (shadow)=197259172
alpha = [1216263864,596674914,215404610,197259172]
##alpha = [, , , ]
##print('perc ch', alpha[1]/alpha[0])
print(f"alpha-0 (no-cloud)={alpha[0]}, alpha-1 (thick-cloud)={alpha[1]}, alpha-2 (thin-cloud)={alpha[2]}, alpha-3 (shadow)={alpha[3]}")
#criterion = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha, gamma = 2, smooth = 1e-5).to(device)


print('opttttttttttt', optimizer.param_groups[0]['lr'])

total_iters = len(trainloader) * epochs
print('totaliters', total_iters)
save_folder = 'saved_models' #where to save the models and training progress
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)
ff=open('./' + save_folder + '/progress.txt','w')
iter_ = 0


save_folder = 'saved_models' #where to save the models and training progress
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)

for epoch in range(1, epochs+1):

    net.train()
    train_losses = []
#    confusion_matrix = tnt.meter.ConfusionMeter(2, normalized=True)
    cm_total = np.zeros((4,4))
    for i, batch in enumerate(tqdm(trainloader)):

        imgs, labels = batch
        
#################################################################################################################################

#        img_save = imgs[0].permute(1,2,0).data.cpu().numpy()
#        lbl_save = labels[0].squeeze().data.cpu().numpy()
        
#        cv2.imwrite('./check/images/img_{}.png'.format(i), img_save[:,:,[2,1,0]]*256)
#        cv2.imwrite('./check/labels/lbl_{}.png'.format(i), lbl_save*256)

#################################################################################################################################


        imgs, labels = imgs.float().to(device), labels.long().to(device)
        optimizer.zero_grad()

        preds = net(imgs)
    
        label_conf, pred_conf = labels.data.cpu().numpy().flatten(), torch.argmax(preds,1).data.cpu().numpy().flatten()
        cm = confusion_matrix(label_conf, pred_conf, labels=[0, 1, 2, 3])
        cm_total += cm
        
        loss = criterion(preds, labels)
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        iter_ += 1
        #lr_scheduler.step(epoch + i / iters)
        #lr_ = base_lr * (1.0 - iter_ / total_iters) ** 0.9
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr_


            
        if iter_ % 20 == 0:
            pred = preds[0]
            pred = torch.softmax(pred, 0)
            pred = np.argmax(pred.data.cpu().numpy(), axis=0)
            gt = labels.data.cpu().numpy()[0]
            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss_CH: {:.6f}\tAccuracy: {}'.format(
                      epoch, epochs, i, len(trainloader),100.*i/len(trainloader), loss.item(), tools.accuracy(pred, gt)))
        
        



    train_acc=(np.trace(cm_total)/float(np.ndarray.sum(cm_total))) *100
    print('TRAIN_LOSS: ', '%.3f' % np.mean(train_losses), 'TRAIN_ACC: ', '%.3f' % train_acc)
    print(cm_total)
    #prec, rec, f1 = tools.metrics(cm_total)
    #print ('Precision: {}\nRecall: {}\nF1: {}'.format(prec, rec, f1)) 
    
    cm_total = np.zeros((4,4))
    with torch.no_grad():
        net.eval()
        val_losses = []

        for i, batch in enumerate(tqdm(valloader)):
            imgs, labels = batch
            imgs, labels = imgs.float().to(device), labels.long().to(device)
#            preds, d0, d1, d2, d3, d4, amp41, amp31, amp21, amp11, amp01 = net(imgs)            
            preds = net(imgs)
            label_conf, pred_conf = labels.data.cpu().numpy().flatten(), torch.argmax(torch.softmax(preds, 1),1).data.cpu().numpy().flatten()
            cm = confusion_matrix(label_conf, pred_conf, labels=[0, 1, 2, 3])
            cm_total += cm

            loss = criterion(preds, labels)
            val_losses.append(loss.item())


        #scheduler.step(np.mean(val_losses))    
        test_acc=(np.trace(cm_total)/float(np.ndarray.sum(cm_total))) *100
        print('VAL_LOSS: ', '%.3f' % np.mean(val_losses), 'VAL_ACC: ', '%.3f' % test_acc)
        print('VALIDATION CONFUSION MATRIX')    
        print(cm_total)
        #prec, rec, f1 = tools.metrics(cm_total)
        #print ('Precision: {}\nRecall: {}\nF1: {}'.format(prec, rec, f1)) 
    

    #tools.write_results(ff, save_folder, epoch, train_acc, test_acc, np.mean(train_losses), np.mean(val_losses), cm_total, prec, rec, f1, optimizer.param_groups[0]['lr'])

    #save model in every epoch
    torch.save(net.state_dict(), './' + save_folder + '/net_{}.pt'.format(epoch))
