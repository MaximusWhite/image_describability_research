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
import sys
import itertools
from itertools import islice
import numpy as np
import matplotlib as mtp
from matplotlib.image import imread
import torch
from PIL import Image
from torch.nn import Linear
import torch.nn as nn 
from torchvision.models.resnet import resnet50

from torchsummary import summary


meta_version = 'v6'



config = {
    'scores_filename': 'total_scores_val_v6.json',
    'img_path': '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco/val2014',
    'path_to_embed': '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_val2014_embed/resnet_img_embeds'
}

transform_pile = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])


with open(os.path.join('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/meta/{}/'.format(meta_version), config['scores_filename']), 'r') as infile:
            dataset = json.load(infile)       

gpu_id = 0 if len(sys.argv) == 1 else int(sys.argv[1])
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
print('cuda name: {}, ID: {}'.format(torch.cuda.get_device_name(gpu_id), gpu_id))
device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else "cpu")

# https://medium.com/the-owl/extracting-features-from-an-intermediate-layer-of-a-pretrained-model-in-pytorch-c00589bda32b

class custom50(nn.Module):
    def __init__(self, output_layer):
        super().__init__()
        self.output_layer = output_layer
        self.pretrained = resnet50(pretrained=True, progress=True)
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == self.output_layer:
                break

        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None
        
    def forward(self,x):
        x = self.net(x)
        return x


model = custom50(output_layer = 'avgpool')
model.to(device)
model.eval()

embed_list = [] 
img_id_list = []

total = len(dataset)
curr = 0
for i in range(len(dataset)):
    filename = os.path.join(config['img_path'], dataset[i]['file_name'])
    image = Image.open(filename)

    if image.mode != 'RGB':  # grayscale -> RGB
        image = image.convert('RGB')
    image = transform_pile(image)
    image = torch.stack([image])
    image = image.to(device=device, dtype=torch.float)
    
    img_id_list.append(dataset[i]['id'])
    
    with torch.no_grad():
        output = model(image)
    embed_list.append(output.cpu().detach().numpy().flatten())
    
    curr += 1 
    
    if curr % 500 == 0:    
        print('\rProcessed {:4.4f}%'.format((curr / total * 100)), end="")
        sys.stdout.flush()
print()
np.save(open(config['path_to_embed'] + "/coco2014val_resnet_img_embeds.npy", "wb"), np.array(list(zip(img_id_list, embed_list))))

    