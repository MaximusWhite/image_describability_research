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
import nltk
# nltk.download('wordnet')
import itertools
import numpy as np
import matplotlib as mtp
from heapq import heapify, heappop, heappush, _heapify_max, _heappop_max
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

nltk.download('punkt')

# import bert_score
# from bert_score import BERTScorer
random.seed(time.perf_counter())

import warnings
warnings.filterwarnings('ignore')

from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.tokenizer.ptbtokenizer import PTBTokenizer

from collections import defaultdict
# prereqs:
# https://github.com/Maluuba/nlg-eval 
from rouge_score import rouge_scorer
# from nlgeval import NLGEval
# nlgeval = NLGEval()
import time


# meta_path =  os.path.expanduser('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/meta/v3/')

path_to_corpus_data = "./coco_caption/data/coco_train2014_corpus.json"
path_to_ref_file = "./coco_caption/data/final_refs.json"
path_to_cand_file = "./coco_caption/data/final_cands.json"
# path_to_ref_file = "./coco_caption/data/coco2014val_refs.json"
# path_to_cand_file = "./coco_caption/data/coco2014val_cands.json"
# path_to_ref_file = "./coco_caption/data/gts.json"
# path_to_cand_file = "./coco_caption/data/cands.json"

# ref_file = "./coco_caption/data/gts.json"
# cand_file = "./coco_caption/data/refs.json"

results_filename = 'cider_final'  ## default: cider_scores_adjusted_map

print('loading data files...', end="")

corpus = json.loads(open(path_to_corpus_data, 'r').read())
ref_list = json.loads(open(path_to_ref_file, 'r').read())
cand_list = json.loads(open(path_to_cand_file, 'r').read())

print('done')

corp = defaultdict(list)
gts = defaultdict(list)
res = defaultdict(list)

porter = PorterStemmer()

def strip_line(line):
    return (line.translate(str.maketrans('', '', string.punctuation))).rstrip()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

print('preprocessing...', end="")

###### FOR DEBUG ONLY ##############################################################

# for l in ref_list:
#     gts[l['image_id']].append(strip_line(l['caption']))

# # change of naming convention from cand to res
# for l in cand_list:
#     res[l['image_id']].append(strip_line(l['caption']))
    
# for l in corpus:
#     corp[l['image_id']].append(strip_line(l['caption']))

####################################################################################

# lancaster = LancasterStemmer()

# change of naming convention from ref to gts
for l in ref_list:
    gts[l['image_id']].append(stemSentence(strip_line(l['caption'])))

# change of naming convention from cand to res
for l in cand_list:
    res[l['image_id']].append(stemSentence(strip_line(l['caption'])))
    
for l in corpus:
    corp[l['image_id']].append(stemSentence(strip_line(l['caption'])))    # UNCOMMENT IF WANT TO USE STEMMER


### WITH TOKENIZER

# for l in ref_list:
#     gts[l['image_id']].append(stemSentence(l['caption']))

# #change of naming convention from cand to res
# for l in cand_list:
#     res[l['image_id']].append(stemSentence(l['caption']))
    
# for l in corpus:
#     corp[l['image_id']].append(stemSentence(l['caption']))

# print('tokenization...')
# tokenizer = PTBTokenizer()

# gts  = tokenizer.tokenize(gts)
# res = tokenizer.tokenize(res)
# corp = tokenizer.tokenize(corp)
print('done')
    
cider_scorer = Cider('custom')

cider_scorer.prepare_df(corp)

# with open('./coco_caption/data/coco_val2014_stem_df.p', 'wb') as f:
#     pickle.dump(cider_scorer.get_df(), f, protocol=pickle.HIGHEST_PROTOCOL)
    
score, scores = cider_scorer.compute_score(gts, res)

cider_map = {}

print('converting into map...', end="")
for sc in scores:
    cider_map[sc['img_id']] = sc['score']
print('done')

print('saving scores...', end="")
with open('./coco_caption/results/{}.json'.format(results_filename), 'w') as f:
#     json.dump(scores.tolist(), f)
    json.dump(cider_map, f)

print('done')

# print(scores)
# print(score)


