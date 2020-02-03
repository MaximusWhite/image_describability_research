from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import os
from scipy import misc
from scipy.interpolate import interp1d
import pandas
import random
import time
import nltk
import itertools
from itertools import islice
import numpy as np
import matplotlib as mtp
from matplotlib.image import imread
import torch
from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, mode='train', split=(0.7, 0.1), transform=None, targets=None):
        self.path_to_coco = os.path.expanduser('~/Projects/image_captioning/datasets/coco/annotations/')
        # self.path_to_salicon = os.path.expanduser('~/Projects/image_captioning/datasets/salicon/')
        self.meta_path = os.path.expanduser('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/meta/v2/')
        self.path_to_img = '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco/train2014/'
        self.targets = targets
        with open(os.path.join(self.meta_path, 'total_scores.json'), 'r') as infile:
            dataset = json.load(infile)

        train_edge = int(np.interp(split[0], [0, 1], [0, len(dataset) - 1]))
        val_edge = train_edge + int(np.interp(split[1], [0, 1], [0, len(dataset) - 1]))

        if mode == 'train':
            dataset = dataset[:train_edge]
        elif mode == 'val':
            dataset = dataset[train_edge:val_edge]
        else:
            dataset = dataset[val_edge:]

        self.mode = mode    # train, val, test
        print('{} size: {}'.format(mode, len(dataset)))
        self.dataset = dataset
        self.transform = transform  # should I transform anything? UPD: yes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = os.path.join(self.path_to_img, self.dataset[idx]['file_name'])
        image = Image.open(filename)

        if image.mode != 'RGB':  # grayscale -> RGB
            image = image.convert('RGB')

        if self.transform:
            # print('transforming {}'.format(image.size), end='')
            image = self.transform(image)
            # print(' to {}'.format(image.shape))
        # print('tshape = ', )

        # if not len(image.shape) == 3:
            # dim = np.zeros((480, 640))
            # image = np.stack((image, dim, dim), axis=2)
            # image = np.resize(image, (480, 640, 3))
        # image = image.transpose((2, 0, 1))
        if self.targets == None:
            scores = [metric['score_average'] for metric in self.dataset[idx]['data']['averages']]
        else:
            scores = [self.dataset[idx]['data']['averages'][metric_ind]['score_average'] for metric_ind in self.targets]
        sample = {
            'img': image,
            'scores': torch.from_numpy(np.array(scores))
        }

        # sample = {
        #     's': image.shape,
        #     'img': image
        # }

        return sample