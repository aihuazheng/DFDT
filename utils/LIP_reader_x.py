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
        if split == "train":
            self.flip = True
        else:
            self.flip = False
            
        if split == 'train':
            self.data_dir = "/data/huiminwang/dataset/LIP/trainval_images/train_images/"
            self.mask_dir = "/data/huiminwang/dataset/LIP/TrainVal_parsing_annotations/train_segmentations/"
            self.data_list = "/data/huiminwang/dataset/LIP/trainval_images/train_id.txt"

        elif split == 'val':
            self.data_dir = "/data/huiminwang/dataset/LIP/trainval_images/val_images/"
            self.mask_dir = "/data/huiminwang/dataset/LIP/TrainVal_parsing_annotations/val_segmentations/"
            self.data_list = "/data/huiminwang/dataset/LIP/trainval_images/val_id.txt"
        else:
            self.data_dir = "/data/huiminwang/dataset/LIP/testing_images/testing_images/"
            self.mask_dir = None
            self.data_list = "/data/huiminwang/dataset/LIP/testing_images/test_id.txt"
        
        self.image_list, self.label_list, self.label_list_rev = \
            read_labeled_image_list(self.data_dir, self.mask_dir, self.data_list)

        # self.data_dir_train = "/data1/xuran/LIP/LIP/train_images/"
        # self.mask_dir_train = "/data1/xuran/LIP/LIP/TrainVal_parsing_annotations/train_segmentations/"
        # self.data_list_train = "/data1/xuran/LIP/LIP/train_id.txt"
        #
        # self.data_dir_val = "/data1/xuran/LIP/LIP/val_images/"
        # self.mask_dir_val = "/data1/xuran/LIP/LIP/TrainVal_parsing_annotations/val_segmentations/"
        # self.data_list_val = "/data1/xuran/LIP/LIP/val_id.txt"
        #
        
        # self.image_list_train, self.label_list_train, self.label_list_rev_train = \
        #     read_labeled_image_list(self.data_dir_train, self.mask_dir_train, self.data_list_train)
        #
        # self.image_list_val, self.label_list_val, self.label_list_rev_val = \
        #     read_labeled_image_list(self.data_dir_val, self.mask_dir_val, self.data_list_val)
        #
        # self.image_list = self.image_list_train + self.image_list_val
        # self.label_list = self.label_list_train + self.label_list_val
        # self.label_list_rev = self.label_list_rev_train + self.label_list_rev_val
        
    
    def _image_loader(self, index):
        
        file_name = self.image_list[index]
        #print("$$$$$$$file_name######"+str(file_name)) #/data/huiminwang/dataset/LIP/trainval_images/train_images/65869_440087
        label_name = self.label_list[index]
        #print("$$$$$$$label_name######"+str(label_name)) #/data/huiminwang/dataset/LIP/TrainVal_parsing_annotations/train_segmentations/65869_440087
        label_name_rev = self.label_list_rev[index]
        #print("$$$$$$$lable_name_rev######"+str(label_name_rev)) #/data/huiminwang/dataset/LIP/TrainVal_parsing_annotations/train_segmentations/65869_440087_rev
        
        image = pil_loader_RGB(file_name + ".jpg")
        label = pil_loader(label_name + ".png")
        if self.flip:
            label_rev = pil_loader(label_name_rev + ".png")
        
            return image, label, label_rev
        else:
            return image, label, label
    
    def __getitem__(self, index):
        
        img, label, label_rev = self._image_loader(index)
        ######################################
        return self._transform(img, label, label_rev)
    
    def __len__(self):
        return len(self.image_list)
    
    def _transform(self, image, label, label_rev):
        
        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label_rev

        image = np.array(image, copy=False)
        label = np.array(label, copy=False)
        label = np.reshape(label, (label.shape[0], label.shape[1], 1))
        
        base_size = self.opt.sample_size  # (h ,w)
        ori_size = image.shape[0:2]  # (h, w)
        
        if self.scale:
            ori_size_h, ori_size_w = ori_size
            # if ori_size_h / base_size[0] > ori_size_w / base_size[1]:
            if ori_size_h > ori_size_w:
                ori_size_h, ori_size_w = base_size[0], int(base_size[0] * ori_size_w / ori_size_h)
            else:
                ori_size_h, ori_size_w = int(base_size[1] * ori_size_h / ori_size_w), base_size[1]
            
            image = cv2.resize(image, (ori_size_w, ori_size_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (ori_size_w, ori_size_h), interpolation=cv2.INTER_NEAREST)
            scale_factor = random.uniform(0.75, 1.25)
            # scale_factor = random.uniform(1, 1)
            scale_size_h, scale_size_w = int(ori_size_h * scale_factor), int(ori_size_w * scale_factor)
        else:
            scale_size_h, scale_size_w = base_size
        
        image = cv2.resize(image, (scale_size_w, scale_size_h), interpolation=cv2.INTER_LINEAR)
        # label = cv2.resize(label, (30, 30), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (scale_size_w, scale_size_h), interpolation=cv2.INTER_NEAREST)
        # label = cv2.resize(label, (scale_size_w, scale_size_h), interpolation=cv2.INTER_NEAREST)
        
        if self.scale:
            pad_h = max(base_size[0] - scale_size_h, 0)
            pad_w = max(base_size[1] - scale_size_w, 0)
            
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,
            }
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=(0.0, 0.0, 0.0), **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=self.opt.ignore_label, **pad_kwargs)
            
            start_h = random.randint(0, max(image.shape[0] - base_size[0], 0))
            start_w = random.randint(0, max(image.shape[1] - base_size[1], 0))
            end_h = start_h + base_size[0]
            end_w = start_w + base_size[1]
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]
        
        # print image.shape, label.shape
        
        # image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = image.float().div(self.opt.norm_value)
        
        for t, m, s in zip(image, self.opt.mean, self.opt.std):
            t.sub_(m).div_(s)
        return image, label


############################################

def read_labeled_image_list(data_dir, mask_dir, data_list):
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
    for line in f:
        try:
            image, mask, mask_rev = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = mask = mask_rev = line.strip("\n")
        images.append(data_dir + image) #-->/data/huiminwang/dataset/LIP/trainval_images/train_images/159331_2164318
        masks.append(mask_dir + mask) #-->/data/huiminwang/dataset/LIP/TrainVal_parsing_annotations/train_segmentations/159331_2164318
        masks_rev.append(mask_dir + mask + "_rev") #-->/data/huiminwang/dataset/LIP/TrainVal_parsing_annotations/train_segmentations/159331_2164318_rev
    return images, masks, masks_rev


def read_pose_list(data_dir, data_id_list):
    f = open(data_id_list, 'r')
    poses = []
    for line in f:
        pose = line.strip("\n")
        poses.append(data_dir + '/heatmap/' + pose)
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
        parser.add_argument('--norm_value', default=1, type=int,
                            help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
        parser.add_argument('--ignore_label', default=0, type=float)
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
    
    for i, (inputs, targets) in enumerate(train_loader):
        print (i)
        if i == 2:
            print (len(inputs))
            print (inputs.shape, targets.shape)
            
            img_1 = inputs[0].numpy()
            print (img_1.shape)
            img_1 = np.transpose(img_1, (1, 2, 0))
            # print (img_1[:, :, 2].shape)
            img_1 = np.stack((img_1[:, :, 2], img_1[:, :, 1], img_1[:, :, 0]), 2)
            # print (img_1.shape)
            # cv2.imshow('img_1', img_1)
            # cv2.waitKey()
            # print targets[0].type(torch.FloatTensor)
            cv2.imwrite("../obselete/test_label.jpg", targets[0].numpy())
            cv2.imwrite("../obselete/test_1.jpg", img_1)
            break
    
    print ('end')
