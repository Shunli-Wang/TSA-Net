import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
from scipy import stats
import pandas as pd
import json
from scipy.io import loadmat
import copy
import xlrd

from opts import *


def load_image_train(image_path, hori_flip, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0, 3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if hori_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    size = input_resize
    interpolator_idx = random.randint(0, 3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = interpolators[interpolator_idx]
    image = image.resize(size, interpolator)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


class VideoDataset(Dataset):

    def __init__(self, mode, args):
        super(VideoDataset, self).__init__()

        self.mode = mode  # train or test
        self.args = args

        # Loading annotations
        if self.mode == 'train':
            self.info = xlrd.open_workbook(os.path.join(self.args.dataset_path, 'Train.xls')).sheet_by_name('Sheet1')
        elif self.mode == 'test':
            self.info = xlrd.open_workbook(os.path.join(self.args.dataset_path, 'Test.xls')).sheet_by_name('Sheet1')

        self.ann = {}
        for i in range(1, self.info.nrows):
            info_list = self.info.row_values(i)
            self.ann[info_list[0]] = info_list[1]

        # keys
        self.keys = list(self.ann.keys())

        # Load Boxes
        with open(os.path.join(self.args.dataset_path, 'FRFS_Boxes.json'), 'r') as file_object:
            self.boxes = json.load(file_object)

    def get_imgs(self, key):

        transform = transforms.Compose([transforms.CenterCrop(H), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])])

        image_list = sorted((glob.glob(os.path.join(os.path.join(self.args.dataset_path, 'Images'), key, '*.jpg'))))
        sample_range = np.arange(0, num_frames)  # [0, 102]

        # Padding frames
        image_list, box = image_list, copy.deepcopy(self.boxes)[key]
        tmp_a, tmp_b = box[0][6], box[0][7]  # swap the first frame (boxes fix the first frame)
        box[0][6], box[0][7] = box[0][4], box[0][5]
        box[0][4], box[0][5] = tmp_a, tmp_b
        box_h = box

        # Temporal augmentation
        if self.mode == 'train':
            temporal_aug_shift = 0 # random.randint(0, self.args.temporal_aug)  # adding: 0~6 [6-wo aug]
            sample_range += temporal_aug_shift
            # padding init frames for 6:
            for i in range(self.args.temporal_aug):  #
                image_list = [image_list[0]] + image_list
                box_h = [box_h[0]] + box_h  # fill to 109
            box_h = box_h[temporal_aug_shift: 103 + temporal_aug_shift]  # cut to 103
        box = np.array(box_h)  # (109, 8)

        # Spatial augmentation
        if self.mode == 'train':
            hori_flip = False
            if hori_flip:  # flip the x-xl
                box[:, 0], box[:, 2], box[:, 4], box[:, 6] = W_img - box[:, 0], W_img - box[:, 2], W_img - box[:, 4], W_img - box[:, 6]

        # load images
        images = torch.zeros(num_frames, C, H, W)
        for j, i in enumerate(sample_range):
            if self.mode == 'train':
                images[j] = load_image_train(image_list[i], hori_flip, transform)
            if self.mode == 'test':
                images[j] = load_image(image_list[i], transform)

        boxes = torch.tensor(box)  # [103,8]

        return images, boxes

    def __getitem__(self, ix):
        key = self.keys[ix]
        data = {}

        data['keys'] = key
        data['video'], data['boxes'] = self.get_imgs(key)
        data['final_score'] = 1 if self.ann[key] > 0.5 else 0
        return data

    def __len__(self):
        sample_pool = len(self.keys)
        return sample_pool
