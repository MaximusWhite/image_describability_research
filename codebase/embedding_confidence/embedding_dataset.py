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

# IMAGE_EMBED_SIZE = 1280    - for efficientNet stuff 
# IMAGE_EMBED_SIZE = 2048    - for resnet embeds 
# CAPTION_EMBED_SIZE = 768 

meta_version = 'v5'

config = {
#     'img_embed_path': '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_train2014_embed/',
    'img_embed_path': '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_train2014_embed/resnet_img_embeds',
    'cap_embed_path': '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_train2014_captions_embed/',
#     'img_embed_filename': 'total_image_embedding_list.npy',
    'img_embed_filename': 'coco2014_resnet_img_embeds.npy',
    'cap_embed_filename': 'coco2014_caption_embeddings.npy',
    'embed_indexing_map_filename': 'embed_indexing_map.json', # placed in img_embed_path
    
    ###### REGULAR - EFFICIENTNET IMG EMBED + BERT CAPTION
#     'predefined_train_filename': 'predefined_dataset_train_no_repeats_QA.npy',
#     'predefined_val_filename': 'predefined_dataset_val_no_repeats_QA.npy',
#     'predefined_test_filename': 'predefined_dataset_test_no_repeats_QA.npy',
    
    ###### WITH RESNET IMAGE EMBEDDINGS + BERT CAPTION
    
#     'predefined_train_filename': 'predefined_dataset_train_resnet_embeds_no_repeats.npy',
#     'predefined_val_filename': 'predefined_dataset_val_resnet_embeds_no_repeats.npy',
#     'predefined_test_filename': 'predefined_dataset_test_resnet_embeds_no_repeats.npy',
    
       ###### WITH REPEATING PAIRS PER IMAGE
#     'predefined_train_filename': 'predefined_dataset_train_repeat3.npy',
#     'predefined_val_filename': 'predefined_dataset_val_repeat3.npy',
#     'predefined_test_filename': 'predefined_dataset_test_repeat3.npy',
    
    'predefined_train_filename': 'predefined_dataset_train_resnet_embeds_repeat3.npy',
    'predefined_val_filename': 'predefined_dataset_val_resnet_embeds_repeat3.npy',
    'predefined_test_filename': 'predefined_dataset_test_resnet_embeds_repeat3.npy',
    
}

class EmbeddingDataset(Dataset):
    def get_samples(self):
        return np.array(self.sample_bulk)
    def set_mode(self, mode):
        # mode: train/val/test
        if mode == 'train':
            self.working_img_dataset = self.train_img_dataset
            self.working_cap_dataset = self.train_cap_dataset
            self.random_threshold = 0.50
        elif mode == 'val':
            self.working_img_dataset = self.val_img_dataset
            self.working_cap_dataset = self.val_cap_dataset
            self.random_threshold = 0.50
        else:
            self.working_img_dataset = self.test_img_dataset
            self.working_cap_dataset = self.test_cap_dataset
            self.random_threshold = 0.50
        
        self.mode = mode
        
        return self
        
    def __init__(self, mode='train', prepicked_pairs=False, split=(0.7, 0.1), random_threshold=0.50):
        
        self.random_threshold = 0.50
        self.image_embed_path = config['img_embed_path']
        self.caption_embed_path = config['cap_embed_path']
        
        self.prepicked_pairs = prepicked_pairs
        
        if self.prepicked_pairs:
            self.mode = mode
            self.working_dataset = np.load(os.path.join(self.image_embed_path, config['predefined_{}_filename'.format(mode)]), allow_pickle=True)
            
        else:
            
            print('USING DYNAMIC DATASET PAIRING')
            self.img_embed_dataset = np.load(os.path.join(self.image_embed_path, config['img_embed_filename']), allow_pickle=True)
            self.cap_embed_dataset = np.load(os.path.join(self.caption_embed_path, config['cap_embed_filename']), allow_pickle=True)

            # INDEXING_MAP: image_id -> list of caption indecies in caption_embed file (for captions belonging to image_id)
            self.indexing_map = json.load(open(os.path.join('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_train2014_embed/', config['embed_indexing_map_filename']), 'r'))

            img_train_edge = int(np.interp(split[0], [0, 1], [0, len(self.img_embed_dataset) - 1]))
            img_val_edge = img_train_edge + int(np.interp(split[1], [0, 1], [0, len(self.img_embed_dataset) - 1]))
            cap_train_edge = int(np.interp(split[0], [0, 1], [0, len(self.cap_embed_dataset) - 1]))
            cap_val_edge = cap_train_edge + int(np.interp(split[1], [0, 1], [0, len(self.cap_embed_dataset) - 1]))
            
            train_img_dataset = self.img_embed_dataset[:img_train_edge]
            train_cap_dataset = self.cap_embed_dataset[:cap_train_edge]

            val_img_dataset = self.img_embed_dataset[img_train_edge:img_val_edge]
            val_cap_dataset = self.cap_embed_dataset[cap_train_edge:cap_val_edge]

            test_img_dataset = self.img_embed_dataset[img_val_edge:]
            test_cap_dataset = self.cap_embed_dataset[cap_val_edge:]

            self.train_img_dataset = train_img_dataset
            self.train_cap_dataset = train_cap_dataset
            self.val_img_dataset = val_img_dataset
            self.val_cap_dataset = val_cap_dataset
            self.test_img_dataset = test_img_dataset
            self.test_cap_dataset = test_cap_dataset

            self.set_mode(mode)
            
#             self.counter = 0
            
#             self.working_cap_index = list(range(len(self.working_cap_dataset)))
#             random.shuffle(self.working_cap_index)
            
            if self.mode == 'train':
                print('train_img size: {}'.format(len(train_img_dataset)))
                print('train_cap size: {}'.format(len(train_cap_dataset)))
            elif self.mode == 'val':
                print('val_img size: {}'.format(len(val_img_dataset)))
                print('val_cap size: {}'.format(len(val_cap_dataset)))
            else:
                print('test_img size: {}'.format(len(test_img_dataset)))
                print('test_cap size: {}'.format(len(test_cap_dataset)))
#         self.sample_bulk = []
        print('Random threshold: {}'.format(self.random_threshold))
    def __len__(self):
        return len(self.working_dataset) if self.prepicked_pairs else len(self.working_img_dataset)

    def __getitem__(self, idx):
        
#         self.counter += 1
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.prepicked_pairs:
            data = self.working_dataset[idx][0]
            gt = np.array([self.working_dataset[idx][1]])
#             meta = self.working_dataset[idx][2]
#             print('embed: {} '.format(type(data)))
#             print('gt: {} '.format(type(gt)))
            
        else:
            # grab an image_embed    
            img_embed = self.working_img_dataset[idx][1]
            img_id = self.working_img_dataset[idx][0]

            true_pairing = True if np.random.uniform(0,1) < self.random_threshold else False

            if true_pairing:
                caption_pool = self.indexing_map[str(img_id)]  # get all the caption ids for drawn image
                random_pick = caption_pool[np.random.randint(0, len(caption_pool))]
                cap_embed = self.cap_embed_dataset[random_pick][2]
            else:
#                 random_pick = self.working_cap_index.pop()
#                 cap_embed = self.working_cap_dataset[random_pick][2]

                  # ORIGINAL
                random_pick = np.random.randint(0, len(self.working_cap_dataset))
                cap_embed = self.working_cap_dataset[random_pick][2]

#                 random_pick = np.random.randint(0, len(self.cap_embed_dataset))
#                 cap_embed = self.cap_embed_dataset[random_pick][2]
                # IF by some miracle randomly picked caption belongs to randomly picked image... treat it as non-randomly picked
                if int(self.cap_embed_dataset[random_pick][0]) == int(img_id):
                    true_pairing = True

            data = np.append(img_embed, cap_embed)
            gt = np.array([1.0 if true_pairing else 0.0])
            
            
#             meta = '{}-{}'.format(self.working_img_dataset[idx][0], self.cap_embed_dataset[random_pick][1])
#         if self.mode == 'test':
#             self.sample_bulk.append([data, gt, meta])
        sample = {
            'data': data,
            'is_true_pair': torch.from_numpy(gt),
#             'meta': meta
        }
        return sample