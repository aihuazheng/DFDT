from __future__ import division
import math
import random
import scipy.misc
import numpy as np
from scipy.stats import multivariate_normal
import scipy.io as sio
import csv
import os
import argparse
import numpy as np
import torch
import random
import torch
import torch.utils.data as data
import pickle as pickle
import numpy as np
import time
from PIL import Image
import os
import math
import functools
import copy
import random
import scipy.io as sio
import cv2

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


if __name__ == '__main__':
    f = open("/data/huiminwang/dataset/LIP/trainval_images/val_id.txt", 'r')
    images = []
    masks = []
    masks_rev = []
    image_name =[]
    for line in f:
        try:
            image, mask, mask_rev = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = mask = mask_rev = line.strip("\n")
        images.append("/data/huiminwang/dataset/LIP/trainval_images/val_images/" + image)
        masks.append("/data/huiminwang/dataset/LIP/TrainVal_parsing_annotations/val_segmentations/" + mask)
        image_name.append(image)
        # masks_rev.append(data_dir + mask_rev)
    
    print (len(masks))

    for i in range(len(masks)):
        label = pil_loader(masks[i] + ".png")
        label_rev = np.array(label.transpose(Image.FLIP_LEFT_RIGHT))

        label_new = np.copy(label_rev)
        
        label_new[label_rev==14] = 15
        label_new[label_rev==15] = 14
        label_new[label_rev==16] = 17
        label_new[label_rev==17] = 16
        label_new[label_rev==18] = 19
        label_new[label_rev==19] = 18
        print (image_name[i])
        cv2.imwrite("/data/huiminwang/dataset/LIP/TrainVal_parsing_annotations/val_segmentations/"+image_name[i]+"_rev.png", label_new)
        
    