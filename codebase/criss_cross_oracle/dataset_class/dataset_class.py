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


meta_version = 'v6'

config = {
    'scores_filename': 'criss_cross_averages.json',
    'img_path': '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco/val2014'
}

class ImageDataset(Dataset):

    def __init__(self, mode='train', split=(0.7, 0.1), transform=None):
        self.meta_path = os.path.expanduser('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/meta/{}/'.format(meta_version))
        self.path_to_img = config['img_path']
        with open(os.path.join(self.meta_path, config['scores_filename']), 'r') as infile:
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
        avg_cc = self.dataset[idx]['avg_cc']
        
        image = Image.open(filename)

        if image.mode != 'RGB':  # grayscale -> RGB
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        sample = {
            'img': image,
            'avg_cc': torch.from_numpy(np.array([avg_cc]))
        }
        return sample