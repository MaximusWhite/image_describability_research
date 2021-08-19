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

config = {
#     'img_embed_path': '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_train2014_embed/',
    'img_embed_path': '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_train2014_embed/resnet_img_embeds',
    'cap_embed_path': '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_train2014_captions_embed/',
#     'img_embed_filename': 'total_image_embedding_list.npy',    # BERT embeds 
    'img_embed_filename': 'coco2014_resnet_img_embeds.npy',
    'cap_embed_filename': 'coco2014_caption_embeddings.npy',
    'embed_indexing_map_filename': 'embed_indexing_map.json' # placed in OG img_embed_path
}


random_threshold = 0.50
split = (0.7, 0.1, 0.2) # train, val, test size (% from total dataset)


img_embed_dataset = np.load(os.path.join(config['img_embed_path'], config['img_embed_filename']), allow_pickle=True)
cap_embed_dataset = np.load(os.path.join(config['cap_embed_path'], config['cap_embed_filename']), allow_pickle=True)
# INDEXING_MAP: image_id -> list of caption indecies in caption_embed file (for captions belonging to image_id)
indexing_map = json.load(open(os.path.join('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_train2014_embed/', config['embed_indexing_map_filename']), 'r'))

train_size = int(split[0] * len(img_embed_dataset))
val_size = int(split[1] * len(img_embed_dataset))
test_size = len(img_embed_dataset) - train_size - val_size

train_dataset = []
train_true = 0
val_dataset = []
val_true = 0
test_dataset = []
test_true = 0

index_list = list(range(len(img_embed_dataset)))
random.shuffle(index_list)

index_keys = list(indexing_map.keys())
random.shuffle(index_keys)

def get_random_key():
    return random.choice(index_keys)


for _ in range(train_size):
    ind = index_list.pop()
    for _ in range(3):             # do the same image n times 
        true_pairing = True if np.random.uniform(0,1) < random_threshold else False
        if true_pairing:
            train_true += 1
        img_embed = img_embed_dataset[ind][1]
        img_id = img_embed_dataset[ind][0]
        
        if true_pairing:
            caption_pool = indexing_map[str(img_id)]  # get all the caption ids for drawn image
            cap_index = np.random.randint(0, len(caption_pool))
            random_pick = caption_pool[cap_index]
            del indexing_map[str(img_id)][cap_index] # remove so can't pick the same caption again
            cap_embed = cap_embed_dataset[random_pick][2]
        else:
            caption_pool = []
            while len(caption_pool) <= 3:
                random_img = get_random_key()
                caption_pool = indexing_map[random_img]  # get all the caption ids for drawn image
            
            cap_index = np.random.randint(0, len(caption_pool))
            random_pick = caption_pool[cap_index]
#             random_pick = np.random.randint(0, len(cap_embed_dataset))
            del indexing_map[random_img][cap_index] # remove so can't pick the same caption again
            cap_embed = cap_embed_dataset[random_pick][2]
            # IF by some miracle randomly picked caption belongs to randomly picked image... treat it as non-randomly picked
#             if cap_embed_dataset[random_pick][0] == img_id:
            if int(random_img) == int(img_id):
                print('CAUGHT UNINTENTIONAL PAIRING')
                true_pairing = True
#         print('img: {}; cap: {}'.format(type(img_embed), type(cap_embed)))
        data = np.append(img_embed, cap_embed)
        sample = (data, 1.0 if true_pairing else 0.0, '{}-{}'.format(img_id, cap_embed_dataset[random_pick][1]))
        train_dataset.append(sample)
        
train_dataset = np.array(train_dataset)
print('Training dataset done')
print('Length: {}'.format(len(train_dataset)))
print('1s distribution: {}'.format(train_true / len(train_dataset)))

for i in range(val_size):
    ind = index_list.pop()
    for k in range(1):             # only one for val
        true_pairing = True if np.random.uniform(0,1) < random_threshold else False
        if true_pairing:
            val_true += 1
        img_embed = img_embed_dataset[ind][1]
        img_id = img_embed_dataset[ind][0]
        
        if true_pairing:
            caption_pool = indexing_map[str(img_id)]  # get all the caption ids for drawn image
            cap_index = np.random.randint(0, len(caption_pool))
            random_pick = caption_pool[cap_index]
            del indexing_map[str(img_id)][cap_index] # remove so can't pick the same caption again
            cap_embed = cap_embed_dataset[random_pick][2]
        else:
            caption_pool = []
            while len(caption_pool) <= 2:
                random_img = get_random_key()
                caption_pool = indexing_map[random_img]  # get all the caption ids for drawn image
            cap_index = np.random.randint(0, len(caption_pool))
            random_pick = caption_pool[cap_index]
#             random_pick = np.random.randint(0, len(cap_embed_dataset))
            del indexing_map[random_img][cap_index] # remove so can't pick the same caption again
            cap_embed = cap_embed_dataset[random_pick][2]
            # IF by some miracle randomly picked caption belongs to randomly picked image... treat it as non-randomly picked
#             if cap_embed_dataset[random_pick][0] == img_id:
            if int(random_img) == int(img_id):
                print('CAUGHT UNINTENTIONAL PAIRING')
                true_pairing = True
        data = np.append(img_embed, cap_embed)
        sample = (data, 1.0 if true_pairing else 0.0, '{}-{}'.format(img_id, cap_embed_dataset[random_pick][1]))
        val_dataset.append(sample)
        
val_dataset = np.array(val_dataset)
print('Validation dataset done')
print('Length: {}'.format(len(val_dataset)))
print('1s distribution: {}'.format(val_true / len(val_dataset)))


for i in range(test_size):
    ind = index_list.pop()
    for k in range(1):             # only one for test 
        true_pairing = True if np.random.uniform(0,1) < random_threshold else False
        if true_pairing:
            test_true += 1
        img_embed = img_embed_dataset[ind][1]
        img_id = img_embed_dataset[ind][0]
        
        if true_pairing:
            caption_pool = indexing_map[str(img_id)]  # get all the caption ids for drawn image
            cap_index = np.random.randint(0, len(caption_pool))
            random_pick = caption_pool[cap_index]
            del indexing_map[str(img_id)][cap_index] # remove so can't pick the same caption again
            cap_embed = cap_embed_dataset[random_pick][2]
        else:
            caption_pool = []
            while len(caption_pool) <= 2:
                random_img = get_random_key()
                caption_pool = indexing_map[random_img]  # get all the caption ids for drawn image
            cap_index = np.random.randint(0, len(caption_pool))
#             random_pick = np.random.randint(0, len(cap_embed_dataset))
            random_pick = caption_pool[cap_index]
            del indexing_map[random_img][cap_index] # remove so can't pick the same caption again
            cap_embed = cap_embed_dataset[random_pick][2]
            # IF by some miracle randomly picked caption belongs to randomly picked image... treat it as non-randomly picked
#             if cap_embed_dataset[random_pick][0] == img_id:
            if int(random_img) == int(img_id):
                print('CAUGHT UNINTENTIONAL PAIRING')
                true_pairing = True
        data = np.append(img_embed, cap_embed)
        sample = (data, 1.0 if true_pairing else 0.0, '{}-{}'.format(img_id, cap_embed_dataset[random_pick][1]))
        test_dataset.append(sample)
        
test_dataset = np.array(test_dataset)
print('Testing dataset done')
print('Length: {}'.format(len(test_dataset)))
print('1s distribution: {}'.format(test_true / len(test_dataset)))

np.save(open(config['img_embed_path'] + "/predefined_dataset_train_resnet_embeds_repeat3.npy", "wb"), train_dataset)
np.save(open(config['img_embed_path'] + "/predefined_dataset_val_resnet_embeds_repeat3.npy", "wb"), val_dataset)
np.save(open(config['img_embed_path'] + "/predefined_dataset_test_resnet_embeds_repeat3.npy", "wb"), test_dataset)
