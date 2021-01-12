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
nltk.download('wordnet')
import itertools
import numpy as np
import matplotlib as mtp
from heapq import heapify, heappop, heappush, _heapify_max, _heappop_max
# import bert_score
from bert_score import BERTScorer
random.seed(time.perf_counter())

import warnings
warnings.filterwarnings('ignore')

from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from collections import defaultdict
# prereqs:
# https://github.com/Maluuba/nlg-eval 
from rouge_score import rouge_scorer
# from nlgeval import NLGEval
# nlgeval = NLGEval()
import time


meta_path =  os.path.expanduser('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/meta/v3/')

path_to_annotations = os.path.expanduser('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco/annotations/captions_train2014.json')

cider_scorer = Cider('coco-val-df')
print('Loading BERT model...')
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

gt_tokenizer = PTBTokenizer(_source='gts')
res_tokenizer = PTBTokenizer(_source='gts')

def loadAnnotationsMeta(annotations_path):
    with open(annotations_path, 'r') as f:
        meta = json.load(f)
    return meta

############## Helper loader functions

def make_image_list(images_meta):
    lst = [{
            'id': i['id'],
            'file_name': i['file_name']
            } for i in images_meta]
    with open(os.path.join(meta_path, 'image_list_v3.json'), 'w') as outfile:
        json.dump(lst, outfile)
    return lst

def make_annotations_map(annotations):
    annotations_map = {}
    for img in annotations:
        if str(img['image_id']) in annotations_map:
             annotations_map[str(img['image_id'])].append({
                    'caption': img['caption'].lower(),
                    'id': img['id']
             })
        else:
            annotations_map[str(img['image_id'])] = [{
                    'caption': img['caption'].lower(),
                    'id': img['id']
            }]
    with open(os.path.join(meta_path, 'annotations_map_v3.json'), 'w') as outfile:
                json.dump(annotations_map, outfile)
    return annotations_map

def make_captions_permutations(args):  # arg 0: list, arg 1: annotations map
    captions_permutation_map = {}
    for i in args[0]:
        captions = args[1][str(i['id'])]
        captions_permutation_map[str(i['id'])] = list(itertools.permutations(captions, 2))
    with open(os.path.join(meta_path, 'captions_permutation_map_v3.json'), 'w') as outfile:
        json.dump(captions_permutation_map, outfile)
    return captions_permutation_map

def filterTrainingScores(args):
    train = [{
                'id': i['id'],
                'file_name': i['file_name'],
                'data': args[1][str(i['id'])]
            } for i in args[0]]
    print('Writing {}...'.format(args[2]))
    with open(os.path.join(meta_path, args[2]), 'w') as outfile:
        json.dump(train, outfile)
    return train

#######################################

def loadMetadata(filename, data, callback, override):
    if not override:
        try:
            with open(os.path.join(meta_path, filename), 'r') as file:
                res = json.load(file)
        except FileNotFoundError:
            print('404, creating new "{}"...'.format(filename))
            res = callback(data) 
    else: 
        res = callback(data)
    return res
    
def splitImageList(imageList, trainPercent): # returns train, test
    separation_point = int(len(imageList) * (trainPercent if type(trainPercent) == float else trainPercent / 100))
    return (imageList[:separation_point], imageList[separation_point:])

def bleu_score(ref, can):
    return nltk.translate.bleu_score.sentence_bleu([ref], can, weights = [1])

def meteor_score(ref, can):
    return nltk.meteor([ref], can)

def bert_score_callback(ref, can): 
    P, R, F1 = bert_scorer.score([can],[ref])
    return F1.data[0].item()

def cider_score_callback(refs, can):
    
    ### USING PTBTokenizer
#     gt_list = [{"image_id": "1",
#       "caption": ref} for ref in refs]

#     cand_list = [{"image_id": "1",
#       "caption": can}]

#     def modify_input(gt_list, cand_list):
#         gts = defaultdict(list)
#         res = defaultdict(list)

#         for l in gt_list:
#             gts[l['image_id']].append({"caption": l['caption']})

#         # change of naming convention from cand to res
#         for l in cand_list:
#             res[l['image_id']].append({"caption": l['caption']})

#         gts  = gt_tokenizer.tokenize(gts)
#         res = res_tokenizer.tokenize(res)
        
#         return gts, res

#     gts, res = modify_input(gt_list, cand_list)

    
    ### MANUAL PUNCTUATION REMOVAL
    gts = {}
    res = {}
    
    def strip_line(line):
        return line.translate(str.maketrans('', '', string.punctuation))
    for key in refs:
        gts[key] = [strip_line(l) for l in refs[key]]
        
    for key in can:
        res[key] = [strip_line(l) for l in can[key]]
    
    
    
    
    score, _ = cider_scorer.compute_score(gts, res)
    return score



metrics = [
    {
        'metric_id': 'b_score',
        'callback': bleu_score
    }, 
    {
        'metric_id': 'rouge2',
        'callback': rouge_scorer.RougeScorer(['rouge2'], use_stemmer=False).score,
        'output_transform': (lambda x: x['rouge2'][2])
    },
    {
        'metric_id': 'rouge3',
        'callback': rouge_scorer.RougeScorer(['rouge3'], use_stemmer=False).score,
        'output_transform': (lambda x: x['rouge3'][2])
    },
    {
        'metric_id': 'rougeL',
        'callback': rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False).score,
        'output_transform': (lambda x: x['rougeL'][2])
    },
    {
        'metric_id': 'meteor',
        'callback': meteor_score
    }
]


metrics_v3 = [
    {
        'metric_id': 'b_score',
        'callback': bleu_score
    }, 
    {
        'metric_id': 'rouge2',
        'callback': rouge_scorer.RougeScorer(['rouge2'], use_stemmer=False).score,
        'output_transform': (lambda x: x['rouge2'][2])
    },
    {
        'metric_id': 'rouge3',
        'callback': rouge_scorer.RougeScorer(['rouge3'], use_stemmer=False).score,
        'output_transform': (lambda x: x['rouge3'][2])
    },
    {
        'metric_id': 'rougeL',
        'callback': rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False).score,
        'output_transform': (lambda x: x['rougeL'][2])
    },
    {
        'metric_id': 'meteor',
        'callback': meteor_score
    }, 
    {
        'metric_id': 'bert',
        'callback': bert_score_callback
    },
    {
        'metric_id': 'cider',
        'callback': cider_score_callback,
        'consensus_needed': True
    }
]

def calculate_scores(args):
    captions_permutation_map = args[0]
    annotations_map = args[3]
    metric_callbacks = args[1]
    scores = {}
    total = len(captions_permutation_map)
    count = 0
    for key in captions_permutation_map:
        print(key)
        perm_obj = {
            'scores': {
            }
        }
        
        # for cider and potentially similar metrics
        og_councel = [c['caption'] for c in annotations_map[str(key)]]
#         print('working on {} image ({}%)...'.format(count+1, (count+1) / total))
        
#         start = time.time()
        for permutation in captions_permutation_map[str(key)]:
            reference = permutation[0]['caption']
            candidate = permutation[1]['caption']
        
        ### for using PTBTokenizer
#             councel = og_councel.copy()
#             councel.remove(candidate)

        ### for manual punctuation removal
            councel = {}
            councel[str(key)] = og_councel.copy()
            councel[str(key)].remove(candidate)
            for metric_index in range(len(metric_callbacks)):
                
                    metric_id = metric_callbacks[metric_index]['metric_id']
                    metric_callback = metric_callbacks[metric_index]['callback']
                    out_transform = None if 'output_transform' not in metric_callbacks[metric_index] else metric_callbacks[metric_index]['output_transform']
                    if 'consensus_needed' in metric_callbacks[metric_index]:
                        cand = {}
                        cand[str(key)] = [candidate]
                        
                        score = metric_callback(councel, cand)
                        exit()
#                         print('cider: {}'.format(score))
                    else:
                        score = metric_callback(reference, candidate)
                    if out_transform != None:
                        score = out_transform(score)
                    if str(permutation[1]['id']) not in perm_obj['scores']:
                        perm_obj['scores'][str(permutation[1]['id'])] = { 
                            'caption': permutation[1]['caption'],
                            'metrics': [{
                                            'metric_id': m['metric_id'],
                                            'results': {
                                                    'scores': []
                                            }
                            } for m in metric_callbacks]
                        }

                    perm_obj['scores'][str(permutation[1]['id'])]['metrics'][metric_index]['results']['scores'].append(score) 
                
                
        average_list = [[] for i in range(len(metric_callbacks))]
        variance_list = [[] for i in range(len(metric_callbacks))]
        std_list = [[] for i in range(len(metric_callbacks))]
        for cap_id in perm_obj['scores']:
            for metric_ind in range(len(metric_callbacks)):
                perm_obj['scores'][str(cap_id)]['metrics'][metric_ind]['results']['average'] = np.mean(perm_obj['scores'][str(cap_id)]['metrics'][metric_ind]['results']['scores'])
                perm_obj['scores'][str(cap_id)]['metrics'][metric_ind]['results']['variance'] = np.var(perm_obj['scores'][str(cap_id)]['metrics'][metric_ind]['results']['scores'])
                perm_obj['scores'][str(cap_id)]['metrics'][metric_ind]['results']['std_deviation'] = np.std(perm_obj['scores'][str(cap_id)]['metrics'][metric_ind]['results']['scores'])
                average_list[metric_ind].append(perm_obj['scores'][str(cap_id)]['metrics'][metric_ind]['results']['average'])    
                variance_list[metric_ind].append(perm_obj['scores'][str(cap_id)]['metrics'][metric_ind]['results']['variance'])
                std_list[metric_ind].append(perm_obj['scores'][str(cap_id)]['metrics'][metric_ind]['results']['std_deviation'])
        perm_obj['averages'] = []
        for metric_ind in range(len(metric_callbacks)):
            perm_obj['averages'].append({
                    'metric_id': metric_callbacks[metric_ind]['metric_id'],
                    'score_average': np.mean(average_list[metric_ind]),
                    'variance_average': np.mean(variance_list[metric_ind]),
                    'std_average': np.mean(std_list[metric_ind]),
            })
        scores[str(key)] = perm_obj.copy()
        
        count += 1
        
        print('\rProcessed {:4.4f}%'.format((count / total * 100)), end="")
        sys.stdout.flush()
    print('\nWriting {}...'.format(args[2]))
    with open(os.path.join(meta_path, args[2]), 'w') as outfile:
        json.dump(scores, outfile)
    return scores

annotationsMeta = loadAnnotationsMeta(path_to_annotations)
total_list = loadMetadata('image_list_v3.json',annotationsMeta['images'],make_image_list, False)
train_image_list, test_image_list = splitImageList(total_list, 80)
annotations_map = loadMetadata('annotations_map_v3.json', annotationsMeta['annotations'], make_annotations_map, False)
captions_permutation_map = loadMetadata('captions_permutation_map_v3.json', (total_list, annotations_map), make_captions_permutations, False)
scores = loadMetadata('scores_vtmp.json', (captions_permutation_map, metrics_v3, 'scores_vtmp.json', annotations_map), calculate_scores, True)
# total_scores = loadMetadata('total_scores_v3.json', (total_list, scores, 'total_scores_v3.json'), filterTrainingScores, False)