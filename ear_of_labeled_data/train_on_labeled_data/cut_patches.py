import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import cv2

def find_black_part(img):
    # Black pixel = all channels equal to 0
    black_mask = np.all(img == [0, 0, 0], axis=-1)

    # Percentage
    black_ratio = black_mask.sum() / black_mask.size * 100
    #print(f"Black pixels: {black_ratio:.2f}%")

    return black_ratio

def tile_img(height, width):
    img_x, img_y = [], []

    for x in range(0, height, step):
        #print('x', x)
        if x+psize<=height:
            img_x.append(x)
        else:
            img_x.append(height-psize)
            break
        if x==0:
            for y in range(0, width, step):
                if y+psize<=width:
                    img_y.append(y)
                else:
                    img_y.append(width-psize)
                    break    
    return img_x, img_y


def pad_to_256_center(img):
    h, w, ch = img.shape
    
    # Amount to pad on each dimension
    pad_h = max(0, 256 - h)
    pad_w = max(0, 256 - w)

    # Split padding equally (top/bottom, left/right)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = np.pad(
        img,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )
    return padded

ids = os.listdir('./previews_dataset/IMAGES/')

#ids = ['145925_29.png'] #(500, 197, 3)

psize = 256
step = 256

for _, id in enumerate(tqdm(ids)):

    img = Image.open('./previews_dataset/IMAGES/{}'.format(id))
    label = Image.open('./previews_dataset/LABELS/{}'.format(id))
    img, label = np.array(img), np.array(label)
#    print(img.shape)
    w, h = img.shape[0], img.shape[1]
    if w<256 or h<256:
        padded_img = pad_to_256_center(img)
        padded_label = pad_to_256_center(np.expand_dims(label,2))
        padded_label = padded_label[:,:,0]
        #print(padded_img.shape)
    else:
        padded_img = img.copy()
        padded_label = pad_to_256_center(np.expand_dims(label,2))
        padded_label = padded_label[:,:,0]

    #cv2.imwrite('check.png', padded_img[:,:,[2,1,0]])

    img_x, img_y = tile_img(padded_img.shape[0], padded_img.shape[1])


    for x in img_x:
        for y in img_y:
            img_patch = padded_img[x:x+psize,y:y+psize,:]
            l_patch = padded_label[x:x+psize,y:y+psize]

#            if img_patch.shape[0]!=256 or img_patch.shape[1]!=256:
#                print(img_patch.shape)

            black_perc = find_black_part(img_patch)
            if black_perc<60:
                img_patch = Image.fromarray(img_patch)
                l_patch = Image.fromarray(l_patch)

                img_patch.save('./previews_dataset/PATCHES/images/{}_{}_{}.png'.format(id[:-4],x,y))
                l_patch.save('./previews_dataset/PATCHES/labels/{}_{}_{}.png'.format(id[:-4],x,y))
