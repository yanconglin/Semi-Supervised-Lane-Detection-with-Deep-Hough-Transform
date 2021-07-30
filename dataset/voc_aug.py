import os
import numpy as np

import torch
from torch.utils.data import Dataset
import random
import cv2

class VOCAugDataSet(Dataset):
    def __init__(self, dataset_path='/tudelft.net/staff-bulk/ewi/insy/VisionLab/yanconglin/dataset/lanes/CULane/list', data_list='train_gt', transform=None, mode='train'):
    # def __init__(self, dataset_path='/home/yclin/cluster/dataset/lanes/CULane/list', data_list='train_gt', transform=None, mode='train'):
        print('dataset_path, data_list', dataset_path, data_list)

        # line: start from 1, not 0
        # driver_23_30frame: 1 - 52857, 52858 in total
        # driver_161_90frame: 52858 - 71090, 18233 in total
        # driver_182_30frame: 71091 - 88880, 17790 in total

        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.img_list = []
            self.img = []
            self.label_list = []
            self.exist_list = []
            for line in f:
                self.img.append(line.strip().split(" ")[0])
                self.img_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[0])
                self.label_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[1])
                self.exist_list.append(np.array([int(line.strip().split(" ")[2]), int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5])]))

        self.img_path = dataset_path
        self.gt_path = dataset_path
        self.transform = transform
        self.is_testing = data_list == 'test_img' # 'val'
        self.is_validation = data_list == 'val_gt' # 'val'
        print('data in total', len(self.img_list), len(self.label_list), len(self.exist_list))


        if data_list == 'train_gt':
            seed = 0
            random.seed(seed)
            random.shuffle(self.img)
            random.seed(seed)
            random.shuffle(self.img_list)
            random.seed(seed)
            random.shuffle(self.label_list)
            random.seed(seed)
            random.shuffle(self.exist_list)
            
            self.num_train = int(len(self.img_list) * 0.01)
            self.num_unlabeled = int(len(self.img_list) * 0.99)
            start = 0
            end = len(self.img_list)
            self.img = self.img[start:end]
            self.img_list = self.img_list[start:end]
            self.label_list = self.label_list[start:end]
            self.exist_list = self.exist_list[start:end]
            print('data', data_list, len(self.img_list), len(self.label_list), len(self.exist_list))

        print('data', data_list, len(self.img_list), len(self.label_list), len(self.exist_list))


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # print('self.img_list',self.img_list)
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx])).astype(np.float32)
        label = cv2.imread(os.path.join(self.gt_path, self.label_list[idx]), cv2.IMREAD_UNCHANGED)
        exist = self.exist_list[idx]
        name = self.img[idx]
        # print('image size', image.shape)
        image = image[240:, :, :]
        label = label[240:, :]
        label = label.squeeze()
        # print('image size crop', image.shape)
        # print('data size', np.max(image),np.min(image))
        if self.transform:
            image, label = self.transform((image, label))
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
            label = torch.from_numpy(label).contiguous().long()
            exist = torch.from_numpy(exist).contiguous().float()
        # if self.is_testing:
        #     return image, label, exist, self.img[idx]
        # else:
        #     return image, label, exist, self.img[idx]
        # return image, label, exist, name
        
        if self.is_testing:
            return image, label, exist, name
            
        if self.is_validation:
            return image, label, exist, False
        if idx >= self.num_train:
            return image, label, exist, True
        else:
            return image, label, exist, False

