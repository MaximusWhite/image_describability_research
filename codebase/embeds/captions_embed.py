import json
import os
import random
import time
import pathlib
import string
import sys
import numpy as np

random.seed(time.perf_counter())

import warnings
warnings.filterwarnings('ignore')


MODE = 'val'    # train or val

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def strip_line(line):
    return (line.translate(str.maketrans('', '', string.punctuation))).rstrip()

path_to_annotations = os.path.expanduser('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco/annotations/captions_{}2014.json'.format(MODE))
path_to_embed = '/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco_{}2014_captions_embed'.format(MODE)
with open(path_to_annotations, 'r') as f:
    captions = (json.load(f))['annotations']

captions_raw = [strip_line(c['caption']) for c in captions]
caption_ids = [c['id'] for c in captions]
image_ids = [c['image_id'] for c in captions]
print('calculating...', end="")
caption_embeddings = model.encode(captions_raw)
print('done')
final_embeds = np.array(list(zip(image_ids, caption_ids, caption_embeddings)))

np.save(open(path_to_embed + "/coco2014val_caption_embeddings.npy", "wb"), final_embeds)