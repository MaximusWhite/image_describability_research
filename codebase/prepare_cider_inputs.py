import json
import os
import scipy.io
import pandas 
import random
import importlib
import time
import pathlib
import string
import sys
# import nltk
# nltk.download('wordnet')
import itertools
import numpy as np
import matplotlib as mtp
from heapq import heapify, heappop, heappush, _heapify_max, _heappop_max
# import bert_score
# from bert_score import BERTScorer
random.seed(time.perf_counter())

import warnings
warnings.filterwarnings('ignore')

from coco_caption.pycocoevalcap.cider.cider import Cider
# from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from collections import defaultdict
# prereqs:
# https://github.com/Maluuba/nlg-eval 
from rouge_score import rouge_scorer
# from nlgeval import NLGEval
# nlgeval = NLGEval()
import time

# v3 becase it's the same for all versions
with open('../../../datasets/meta/v3/annotations_map_v3.json', 'r') as f:
# with open('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/meta/abstract50_v1/annotations_map.json', 'r') as f:
    ann_map = json.load(f)
    
with open('../../../datasets/meta/v3/captions_permutation_map_v3.json', 'r') as f:
    captions_permutations_map = json.load(f)
    
# with open('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/meta/v6/annotations_map_val.json', 'r') as f:
#     ann_map = json.load(f)
    
# with open('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/meta/v6/captions_permutation_map_val_v6.json', 'r') as f:
#     captions_permutations_map = json.load(f)
    
class CaptionConsensusPermutator:
    def __init__(self, ann_map):
        self.caption_map = ann_map
    
    def permute(self):
        refs = []
        cands = []
        total = len(self.caption_map)
        count = 0
        for img_id in self.caption_map:
            captions = self.caption_map[img_id]
            for i, caption in enumerate(captions):
                main_id = '{}-{}'.format(img_id, caption['id'])
                cands.append({
                    'image_id': main_id,
                    'caption': caption['caption']
                })
                
                for k, caption_ref in enumerate(captions):
                    if k != i: 
                        refs.append({
                            'image_id': main_id,
                            'caption': caption_ref['caption']
                        })
            count += 1
            print('\rProcessed {:4.4f}%'.format((count / total * 100)), end="")
            sys.stdout.flush()
        with open('./coco_caption/data/final_refs.json', 'w') as f:
            json.dump(refs, f)
        with open('./coco_caption/data/final_cands.json', 'w') as f:
            json.dump(cands, f)
        return (refs, cands)
    
    def permute1v1(self, perm_map):
        refs = []
        cands = []
        total = len(perm_map)
        count = 0
        for img_id in perm_map:
            permutations = perm_map[img_id]
            for perm in permutations: 
                reference = perm[0]['caption']
                candidate = perm[1]['caption']
                # format: img_id - candidate - reference
                id_to_match = '{}-{}-{}'.format(img_id, perm[1]['id'], perm[0]['id'])
                refs.append({'image_id': id_to_match, 'caption': reference})
                cands.append({'image_id': id_to_match, 'caption': candidate})
            count += 1
            print('\rProcessed {:4.4f}%'.format((count / total * 100)), end="")
        with open('./coco_caption/data/final_refs_1v1.json', 'w') as f:
            json.dump(refs, f)
        with open('./coco_caption/data/final_cands_1v1.json', 'w') as f:
            json.dump(cands, f)
        return (refs, cands)

permutator = CaptionConsensusPermutator(ann_map)
# refs, candidaters = permutator.permute()
refs, candidaters = permutator.permute()
refs, candidaters = permutator.permute1v1(captions_permutations_map)