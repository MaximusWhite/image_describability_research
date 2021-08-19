from PIL import Image
import os
from os.path import isfile, join
import json
import sys

MODE = 'val'  # train or val

path_to_images = '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco/{}2014'.format(MODE)
path_to_resized = '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_{}2014_resized'.format(MODE)

path_to_image_list = '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_{}_file_list.json'.format(MODE)

if path_to_image_list is not None:
    with open(path_to_image_list, 'r') as f:
        image_filenames = json.load(f)
else:
    image_filenames = [f for f in os.listdir(path_to_images) if isfile(join(path_to_images, f))] 
    with open('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_{}_file_list.json'.format(MODE), 'w') as f:
        json.dump(image_filenames, f)

total = len(image_filenames)
count = 0

for filename in image_filenames:
    img = Image.open(join(path_to_images, filename))
    if img.mode != 'RGB':  # grayscale -> RGB
            img = img.convert('RGB')
    (img.resize((640, 640))).save(join(path_to_resized, filename))
    
    count += 1
    if count % 100 == 0:
        print('\rProcessed {:4.4f}%'.format((count / total * 100)), end="")
        sys.stdout.flush()
