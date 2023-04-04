import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import scipy.io as sio
import cv2
#from Make_Datasets.Make_Dataset_PA100k import PA100k
from Make_Datasets.Make_Dataset_RAP import RAP
from Make_Datasets.Make_Dataset_PETA import PETA
import random
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def pil_loader_s(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print("######path#####"+str(path))
        with Image.open(f) as img:
            return img.convert('L')


class AttributeDataset(data.Dataset):

    def __init__(self, datasetID, opt, split='test', splitID=0):

        self.datasetID = datasetID
        self.data = {}
        self.split = split
        self.opt = opt
        # self.spatial_transform = spatial_transform

        if self.datasetID == 'PA100k':
            self.db = sio.loadmat(os.path.join('/data/huimin.wang/dataset/PA100k/annotation/', 'annotation.mat'))
            if split == 'train':
                self.data['filenames'] = self.db['train_images_name']
                self.data['labels'] = self.db['train_label']
            elif split == 'test':
                self.data['filenames'] = self.db['test_images_name']
                self.data['labels'] = self.db['test_label']
            elif split == 'val':
                self.data['filenames'] = self.db['val_images_name']
                self.data['labels'] = self.db['val_label']
            self.root_path = '/data/huimin.wang/dataset/PA100k/release_data/'


        elif datasetID == 'RAP':
            self.db = RAP('/data1/shaoheng/whm/dataset/RAP/V1/', splitID)
            if split == 'train':
                self.data['inds'] = self.db.train_ind
                # self.data['labels'] = self.db.train_ind
            elif split == 'test':
                self.data['inds'] = self.db.test_ind
                # self.data['labels'] = self.db.test_ind
            else:
                return None

            self.root_path = '/data1/shaoheng/whm/dataset/RAP/V1/RAP_dataset'


        elif datasetID == 'PETA':
            self.db = PETA(splitID)
            if split == 'train':
                self.data['inds'] = self.db.train_ind

            elif split == 'test':
                self.data['inds'] = self.db.test_ind

            elif split == 'val':
                self.data['inds'] = self.db.val_ind

            self.root_path = '/data1/shaoheng/whm/dataset/PETA/images/'
        else:
            print ('Unknown datasets')


    def _image_loader(self, index):

        if self.datasetID == 'PA100k':
            file_name = self.data['filenames'][index][0][0]
            # print("####filename######"+str(file_name))
            label = self.data['labels'][index]


        elif self.datasetID == 'RAP':
            file_name = self.db._img_names[self.data['inds'][index]][0][0]
            label = self.db.labels[self.data['inds'][index]]

        else:
            file_name = self.db._img_names[self.data['inds'][index]]
            label = self.db.labels[self.data['inds'][index]]


        # print os.path.join(self.root_path, file_name)
        # 按照visual一样修改读取图像
        # image = pil_loader(os.path.join(self.root_path, file_name))
        image_path = os.path.join(self.root_path, file_name)
        image = cv2.imread(image_path, 1)[:, :, ::-1]

        return image, label,file_name


    def __getitem__(self, index):
        img, label ,file_name = self._image_loader(index)


        image = self._transform(img)

        ######################################
        ######################################

        return image, label, file_name


    def __len__(self):
        if 'filenames' in self.data.keys():
            return len(self.data['filenames'])
        else:
            return len(self.data['inds'])


    def _transform(self, image):

        # image = np.array(image, copy=False)

        image = cv2.resize(image, (224, 224))

        image = np.float32(image) / 255

        # input_tensor = preprocess_image(image, mean=[0.5, 0.5, 0.5],
        #                                 std=[0.5, 0.5, 0.5])


        # image = torch.from_numpy(image.transpose((2, 0, 1)))
        # image = image.float().div(self.opt.norm_value)

        # for t, m, s in zip(image, self.opt.mean, self.opt.std):
        #     t.sub_(m).div_(s)

        return image


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    norm_value = [1, 1, 1]
    mean = [0, 0, 0]
    batch_size = 1
    import argparse
    def parse_opts():
        parser = argparse.ArgumentParser()
        parser.add_argument('--sample_size', default=[256, 256], type=int, help='Height and width of inputs')
        parser.add_argument('--mean', default=[0.0, 0.0, 0.0], type=float, help='Weight Decay')
        parser.add_argument('--std', default=[1.0, 1.0, 1.0], type=float, help='Weight Decay')
        parser.add_argument('--norm_value', default=1, type=int,
                            help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
        parser.add_argument('--ignore_label', default=0, type=float)
        args = parser.parse_args()
        return args
    opts = parse_opts()
    training_data = AttributeDataset(datasetID='PETA', opt = opts, split='train', splitID=1)

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=2,
                                               shuffle=False, num_workers=1, pin_memory=True)

    print (training_data.__len__())

    n_classes = 8

    label_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 255),
                    (0, 0, 225), (255, 0, 225), (225, 140, 85), (225, 225, 0)]
    def decode_labels(mask):
        n, h, w, c = mask.shape
        outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
        for i in range(n):
            img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
            pixels = img.load()
            for j_, j in enumerate(mask[i, :, :, 0]):
                for k_, k in enumerate(j):
                    if k < n_classes:
                        pixels[k_, j_] = label_colors[k]
            outputs[i] = np.array(img)
        return outputs



    for i, (inputs, targets, label_parsing) in enumerate(train_loader):
        # print i
        if i == 0:
            print (inputs.shape, targets.shape)

            img_1 = inputs[0].numpy()
            img_1 = np.transpose(img_1, (1, 2, 0))

            # print (img_1[:, :, 2].shape)
            img_1 = np.stack((img_1[:, :, 2], img_1[:, :, 1], img_1[:, :, 0]), 2)
            # print (img_1.shape)
            # cv2.imshow('img_1', img_1)
            # cv2.waitKey()
            # print targets[0].type(torch.FloatTensor)
            # labels_parsing = decode_labels(label_parsing)
            cv2.imshow("../obselete/mmm.jpg", img_1)
            # cv2.imshow("../obselete/aaa.jpg", labels_parsing[0])
        break

    print ('end')