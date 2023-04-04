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


def pil_loader_RGB(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class LIPDataset(data.Dataset):
    def __init__(self, opt, split='test'):
        
        # self.spatial_transform = spatial_transform
        self.opt = opt
        self.scale = False
        self.split = split
        if split == 'train':
            self.flip = True
        else:
            self.flip = False
        
        if split == 'train':
            self.data_dir = "/data/huiminwang/dataset/LIP/trainval_images/train_images/"
            self.mask_dir = "/data/huiminwang/dataset/LIP/TrainVal_parsing_annotations/train_segmentations/"
            self.data_list = "/data/huiminwang/dataset/LIP/trainval_images/train_id.txt"
            self.heat_map_dir = "/data/huiminwang/dataset/LIP/heat_map/train/"
        
        elif split == 'val':
            self.data_dir = "/data/huiminwang/dataset/LIP/trainval_images/val_images/"
            self.mask_dir = "/data/huiminwang/dataset/LIP/TrainVal_parsing_annotations/val_segmentations/"
            self.data_list = "/data/huiminwang/dataset/LIP/trainval_images/val_id.txt"
            self.heat_map_dir = "/data/huiminwang/dataset/LIP/heat_map/val/"
            
        else:
            self.data_dir = "/data/huiminwang/dataset/LIP/testing_images/testing_images/"
            self.mask_dir = None
            self.data_list = "/data/huiminwang/dataset/LIP/testing_images/test_id.txt"
        
        self.image_list, self.label_list, self.label_list_rev, self.heat_map_list = \
            read_labeled_image_list(self.data_dir, self.mask_dir, self.data_list, self.heat_map_dir)
    
    def _image_loader(self, index):
        
        file_name = self.image_list[index]
        label_name = self.label_list[index]
        label_name_rev = self.label_list_rev[index]
        heat_map_name = self.heat_map_list[index]

        
        image = pil_loader_RGB(file_name + ".jpg")
        label = pil_loader(label_name + ".png")
        # label_rev = pil_loader(label_name_rev + ".png")
        pose = [None]*16
        if self.split == 'train':
            for i in range(16):
                pose_i = pil_loader(heat_map_name + '_{}.png'.format(i))
                pose[i] = pose_i
        else:
            for i in range(16):
                pose_i = pil_loader(label_name + ".png")
                pose[i] = pose_i
                
        if self.flip:
            label_rev = pil_loader(label_name_rev + ".png")
    
            return image, label, label_rev, pose
        else:
            return image, label, label, pose
        # return image, label, label_rev, pose
    
    def __getitem__(self, index):
        
        img, label, label_rev, pose  = self._image_loader(index)
        ######################################
        return self._transform(img, label, label_rev, pose)
    
    def __len__(self):
        return len(self.image_list)
    
    def _transform(self, image, label, label_rev, pose):
        
        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label_rev
                
                for k in range(len(pose)):
                    pose[k] = pose[k].transpose(Image.FLIP_LEFT_RIGHT)

                pose_rev = [None] * 16
                pose_rev[0] = pose[5]
                pose_rev[1] = pose[4]
                pose_rev[2] = pose[3]
                pose_rev[3] = pose[2]
                pose_rev[4] = pose[1]
                pose_rev[5] = pose[0]
                pose_rev[10] = pose[15]
                pose_rev[11] = pose[14]
                pose_rev[12] = pose[13]
                pose_rev[13] = pose[12]
                pose_rev[14] = pose[11]
                pose_rev[15] = pose[10]
                pose_rev[6] = pose[6]
                pose_rev[7] = pose[7]
                pose_rev[8] = pose[8]
                pose_rev[9] = pose[9]

                pose = pose_rev
        
        
        image = np.array(image, copy=False)
        label = np.array(label, copy=False)
        label = np.reshape(label, (label.shape[0], label.shape[1], 1))

        pose_pile =[]
        for k in range(len(pose)):
            pose_pile.append(np.array(pose[k], copy=False))
        
        # print pose_pile[0].shape
        
        base_size = self.opt.sample_size  # (h ,w)
        ori_size = image.shape[0:2]  # (h, w)
        

        scale_size_h, scale_size_w = base_size
        
        image = cv2.resize(image, (scale_size_w, scale_size_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (scale_size_w, scale_size_h), interpolation=cv2.INTER_NEAREST)
        
        for k in range(len(pose_pile)):
            pose_pile[k] = cv2.resize(pose_pile[k], (30, 30), interpolation=cv2.INTER_NEAREST)

        pose = np.stack(pose_pile, axis=0)

        

        label = torch.from_numpy(label)
        pose = torch.from_numpy(pose).float().div(self.opt.norm_value)
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = image.float().div(self.opt.norm_value)
        
        for t, m, s in zip(image, self.opt.mean, self.opt.std):
            t.sub_(m).div_(s)
        
        return image, label, pose


############################################

def read_labeled_image_list(data_dir, mask_dir, data_list, heat_map_dir):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    masks_rev = []
    heat_map = []
    for line in f:
        try:
            image, mask, mask_rev = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = mask = mask_rev = line.strip("\n")
        images.append(data_dir + image)
        masks.append(mask_dir + mask)
        masks_rev.append(mask_dir + mask + "_rev")
        heat_map.append(heat_map_dir + image)

    return images, masks, masks_rev, heat_map


def read_pose_list(data_dir, data_id_list):
    f = open(data_id_list, 'r')
    poses = []
    for line in f:
        pose = line.strip("\n")
        poses.append(data_dir + '/heat_map/' + pose)
    return poses


if __name__ == '__main__':
    # images, masks = read_labeled_image_list("/data1/xuran/LIP/LIP/train_images/", \
    #                                         "/data1/xuran/LIP/LIP/TrainVal_parsing_annotations/train_segmentations/",
    #                                         "/data1/xuran/LIP/LIP/train_id.txt")
    
    
    # print images[0], masks[0]
    # print len(images)
    def parse_opts():
        parser = argparse.ArgumentParser()
        parser.add_argument('--sample_size', default=[512, 512], type=int, help='Height and width of inputs')
        parser.add_argument('--mean', default=[0.0, 0.0, 0.0], type=float, help='Weight Decay')
        parser.add_argument('--std', default=[1, 1, 1], type=float, help='Weight Decay')
        parser.add_argument('--norm_value', default=255, type=int,
                            help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
        parser.add_argument('--ignore_label', default=-1, type=float)
        args = parser.parse_args()
        return args
    
    
    opts = parse_opts()
    
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    norm_value = [1, 1, 1]
    mean = [0, 0, 0]
    batch_size = 16
    # spatial_transform = Compose([Scale([512, 512]),
    #                              RandomHorizontalFlip(),
    #                              ToTensor(1),
    #                              Normalize(mean, norm_value),
    #                             ])
    
    training_data = LIPDataset(opt=opts, split='train')
    
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=1,
                                               shuffle=True, num_workers=1, pin_memory=True)
    
    print (training_data.__len__())
    
    for i, (inputs, targets, heat_map) in enumerate(train_loader):
        print (i)
        if i == 2:
            print (len(inputs))
            print (inputs.shape, targets.shape, heat_map.shape)
            
            img_1 = inputs[0].numpy()
            print (img_1.shape)
            img_1 = np.transpose(img_1, (1, 2, 0))
            # print (img_1[:, :, 2].shape)
            img_1 = np.stack((img_1[:, :, 2], img_1[:, :, 1], img_1[:, :, 0]), 2)
            # print (img_1.shape)
            # cv2.imshow('img_1', img_1)
            # cv2.waitKey()
            # print targets[0].type(torch.FloatTensor)
            cv2.imwrite("../obselete/test.jpg",255* img_1)
            for i in range(16):
                cv2.imwrite("../obselete/test_label" + '_{}.png'.format(i), 255*heat_map[0][i].numpy())
            break
    
    print ('end')