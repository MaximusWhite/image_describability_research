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


meta_path =  os.path.expanduser('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/meta/v3/')

path_to_annotations = os.path.expanduser('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco/annotations/captions_train2014.json')

path_to_corpus_data = "./coco_caption/data/coco_train2014_corpus.json"
path_to_ref_file = "./coco_caption/data/coco_train2014_corpus_eval_ref.json"
path_to_cand_file = "./coco_caption/data/coco_train2014_corpus_eval_test.json"

ref_file = "./coco_caption/data/gts.json"
cand_file = "./coco_caption/data/refs.json"

print('loading data files...', end="")

corpus = json.loads(open(path_to_corpus_data, 'r').read())
ref_list = json.loads(open(path_to_ref_file, 'r').read())
cand_list = json.loads(open(path_to_cand_file, 'r').read())

print('done')

corp = defaultdict(list)
gts = defaultdict(list)
res = defaultdict(list)

def strip_line(line):
    return line.translate(str.maketrans('', '', string.punctuation))
    
print('preprocessing...', end="")
# # change of naming convention from ref to gts
for l in ref_list:
    gts[l['image_id']].append(strip_line(l['caption']))

# change of naming convention from cand to res
for l in cand_list:
    res[l['image_id']].append(strip_line(l['caption']))
    
for l in corpus:
    corp[l['image_id']].append(strip_line(l['caption']))

print('done')
    
cider_scorer = Cider('custom')

# print(gts)
# print(res)

cider_scorer.prepare_df(corp)
    
score, scores = cider_scorer.compute_score(gts, res)


print('saving scores...', end="")
with open('./coco_caption/results/cider_scores_adjusted.json', 'w') as f:
    json.dump(scores.tolist(), f)

print('done')

# print(scores)
# print(score)


