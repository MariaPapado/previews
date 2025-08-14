import os
import glob
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
#import torchnet as tnt
from skimage import io


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def shuffle(vector):
  vector = np.asarray(vector)
  p=np.random.permutation(len(vector))
  vector=vector[p]
  return vector

def conf_m(output, target_th):

  output_conf=((output.data.squeeze()).transpose(1,3)).transpose(1,2) #(64,32,32,2)
  output_conf=(output_conf.contiguous()).view(output_conf.size(0)*output_conf.size(1)*output_conf.size(2), output_conf.size(3)) #(65536,2) <-
  target_conf=target_th.data.squeeze() #(64,32,32)
  target_conf=(target_conf.contiguous()).view(target_conf.size(0)*target_conf.size(1)*target_conf.size(2))  #(65536,) <-
  return output_conf, target_conf

def train_append(train_areas, FOLDER_IMAGES, FOLDER_GTS):
    train_before, train_after, train_labels = [], [], []

    for train_id in (train_areas):
        train_before.append(np.load(FOLDER_IMAGES+ '{}/{}_1.npy'.format(train_id, train_id)))
#        train_middle1.append(np.load(FOLDER_IMAGES+ '{}/{}_15.npy'.format(train_id, train_id)))
        train_after.append(np.load(FOLDER_IMAGES+ '{}/{}_2.npy'.format(train_id, train_id)))
        labels = io.imread(FOLDER_GTS+ '{}/cm/{}-cm.tif'.format(train_id, train_id))
        labels[labels==1]=0
        labels[labels==2]=1
        train_labels.append(labels)

    return train_before, train_after, train_labels

def val_append(val_areas, FOLDER_IMAGES, FOLDER_GTS):

    val_before, val_after, val_labels = [], [], []

    for val_id in (val_areas):
        val_before.append(np.load(FOLDER_IMAGES+ '{}/{}_1.npy'.format(val_id, val_id)))
#        val_middle1.append(np.load(FOLDER_IMAGES+ '{}/{}_15.npy'.format(val_id, val_id)))
        val_after.append(np.load(FOLDER_IMAGES+ '{}/{}_2.npy'.format(val_id, val_id)))
        labels = io.imread(FOLDER_GTS+ '{}/cm/{}-cm.tif'.format(val_id, val_id))
        labels[labels==1]=0
        labels[labels==2]=1
        val_labels.append(labels)


    return val_before,  val_after, val_labels

def list_to_array_to_var(before, after, targets):

      before=np.asarray(before)
#      middle1=np.asarray(middle1)
      after=np.asarray(after)

      before=np.reshape(before, (1, before.shape[0], before.shape[1], before.shape[2], before.shape[3]))
#      middle1=np.reshape(middle1, (1, middle1.shape[0], middle1.shape[1], middle1.shape[2], middle1.shape[3]))
      after=np.reshape(after, (1, after.shape[0], after.shape[1], after.shape[2], after.shape[3]))

      batch = np.concatenate( (before, after), 0)
      batch_th = torch.from_numpy(batch).cuda(1)

      targets=torch.from_numpy(np.asarray(targets)).cuda(1)

      return batch_th, targets


def apend(inputs_before, inputs_after, targets, xys_line, patch_size, train_before, train_after, train_labels):
    if xys_line[3]==8000:
      input_before=train_before[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size, :]
      input_before=np.transpose(input_before, (2,0,1))
      inputs_before.append(input_before)

#      input_middle1=train_middle1[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size, :]
#      input_middle1=np.transpose(input_middle1, (2,0,1))
#      inputs_middle1.append(input_middle1)

      input_after=train_after[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size, :]
      input_after=np.transpose(input_after, (2,0,1))
      inputs_after.append(input_after)

      target=train_labels[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size]
      targets.append(target)

    elif xys_line[3]==8001:
      #flip0
      input_before=np.flip( train_before[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size, :], 0)
      input_before=np.transpose(input_before, (2,0,1))
      inputs_before.append(input_before)

#      input_middle1=np.flip( train_middle1[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size, :], 0)
#      input_middle1=np.transpose(input_middle1, (2,0,1))
#      inputs_middle1.append(input_middle1)

      input_after=np.flip(train_after[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size, :], 0)
      input_after=np.transpose(input_after, (2,0,1))
      inputs_after.append(input_after)

      target=np.flip(train_labels[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size], 0)
      targets.append(target)

    elif xys_line[3]==8002:
      #flip1
      input_before=np.transpose( train_before[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size, :], (1,0,2))
      input_before=np.transpose(input_before, (2,0,1))
      inputs_before.append(input_before)

#      input_middle1=np.transpose( train_middle1[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size, :], (1,0,2))
#      input_middle1=np.transpose(input_middle1, (2,0,1))
#      inputs_middle1.append(input_middle1)

      input_after=np.transpose(train_after[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size, :], (1,0,2))
      input_after=np.transpose(input_after, (2,0,1))
      inputs_after.append(input_after)

      target=np.transpose(train_labels[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size], (1,0))
      targets.append(target)

      #flip2
    elif xys_line[3]==8003:
      input_before=np.flip( train_before[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size, :], 1)
      input_before=np.transpose(input_before, (2,0,1))
      inputs_before.append(input_before)

#      input_middle1=np.flip( train_middle1[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size, :], 1)
#      input_middle1=np.transpose(input_middle1, (2,0,1))
#      inputs_middle1.append(input_middle1)

      input_after=np.flip(train_after[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size, :], 1)
      input_after=np.transpose(input_after, (2,0,1))
      inputs_after.append(input_after)

      target=np.flip(train_labels[xys_line[2]-7000] [xys_line[0]:xys_line[0]+patch_size, xys_line[1]:xys_line[1]+patch_size], 1)
      targets.append(target)

    return inputs_before, inputs_after, targets
