import torch
import numpy as np
from torchvision.models import resnet as rn
from torchvision.transforms import transforms
from torch.nn import functional as F
from embedding_dataset import EmbeddingDataset
from torch.utils.data import DataLoader
from torch.nn import Linear, MSELoss, L1Loss, BCEWithLogitsLoss, BCELoss
import torch.nn as nn
import time
import copy
import sys
import os
import json
import shutil
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import random
from confidence_network import SemaConfNet as semaconfnet
from confidence_network import SemaConfNetSplit as semaconfnetsplit

gpu_id = 0 if len(sys.argv) == 1 else int(sys.argv[1])
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
print('cuda name: {}, ID: {}'.format(torch.cuda.get_device_name(gpu_id), gpu_id))
device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else "cpu")
    


# img_embed_list = np.load('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_val2014_embed/total_image_embedding_list.npy', allow_pickle=True)

img_embed_list = np.load('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_val2014_embed/resnet_img_embeds/coco2014val_resnet_img_embeds.npy', allow_pickle=True)

cap_embed_list = np.load('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_val2014_captions_embed/coco2014val_caption_embeddings.npy', allow_pickle=True)

with open('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco/annotations/captions_val2014.json', 'r') as f:
    img_list = json.load(f)
    
id_to_img_embed_map_val = {}
for img in img_embed_list:
    id_to_img_embed_map_val[img[0]] = img[1]

id_to_cap_embed_map_val = {}
for cap in cap_embed_list:
    id_to_cap_embed_map_val[cap[1]] = cap[2]

criss_cross_validation = pd.read_csv('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/criss_cross_data/sits_val_processed.csv')

model = semaconfnetsplit()
model = model.to(device)
model.load_state_dict(torch.load(os.path.join('/mnt/zeta_share_1/mkorchev/image_captioning/models/SemaConfNet/v10-resnet_embeds-DD_final', 'v10-resnet_embeds-DD_final SemaConfNetSplit (lr=1e-6,wd=1e-6)_ACC-0.8999758395747766.config'), map_location=device))
model.eval()

total = len(criss_cross_validation)
count = 0

with torch.no_grad():
    for index, row in criss_cross_validation.iterrows():
        img_id = row['image_id']
        cap_id = row['caption_id']
        img_embed = id_to_img_embed_map_val[img_id]
        cap_embed = id_to_cap_embed_map_val[cap_id]
        embed = np.append(img_embed, cap_embed)
        output = model(torch.from_numpy(np.expand_dims(embed, axis=0)).to(device=device, dtype=torch.float))
        criss_cross_validation.loc[criss_cross_validation['image_id'] == img_id, ['semaconfnet_result']] = output.cpu().detach().numpy()[0].item()
        count += 1
        print('\rProcessed {:4.4f}%'.format((count / total * 100)), end="")
        
criss_cross_validation.to_csv('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/criss_cross_data/sits_val_final.csv', index=False)
    
