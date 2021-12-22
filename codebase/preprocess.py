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
# from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from collections import defaultdict
# prereqs:
# https://github.com/Maluuba/nlg-eval 
from rouge_score import rouge_scorer
# from nlgeval import NLGEval
# nlgeval = NLGEval()


meta_path =  os.path.expanduser('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/meta/v6/')

path_to_annotations = os.path.expanduser('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/coco/annotations/captions_val2014.json')
# path_to_annotations = os.path.expanduser('/mnt/zeta_share_1/mkorchev/image_captioning/datasets/abstract50/annotations/abstract_annotations.json')


# cider_scorer = Cider('coco-val-df')

with open('./coco_caption/results/cider_scores_val_map_1v1.json', 'r') as f:
    cider_score_map = json.load(f)

    
# same format as with cider map: {image_id}-{cand_caption_id}
with open('./coco_caption/data/ref_council_map_val.json', 'r') as f:
    council_map = json.load(f)
    
print('Loading BERT model...')
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

# gt_tokenizer = PTBTokenizer(_source='gts')
# res_tokenizer = PTBTokenizer(_source='gts')

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
    with open(os.path.join(meta_path, 'image_list_val.json'), 'w') as outfile:
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
    with open(os.path.join(meta_path, 'annotations_map_val.json'), 'w') as outfile:
                json.dump(annotations_map, outfile)
    return annotations_map

def make_captions_permutations(args):  # arg 0: list, arg 1: annotations map
    captions_permutation_map = {}
    for i in args[0]:
        captions = args[1][str(i['id'])]
        captions_permutation_map[str(i['id'])] = list(itertools.permutations(captions, 2))
    with open(os.path.join(meta_path, 'captions_permutation_map_val_v6.json'), 'w') as outfile:
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
    return nltk.translate.bleu_score.sentence_bleu(ref, can, weights = [1]) if type(ref) is type([]) else nltk.translate.bleu_score.sentence_bleu([ref], can, weights = [1])

def meteor_score(ref, can):
    return nltk.meteor(ref, can) if type(ref) is type([]) else nltk.meteor([ref], can)

def bert_score_callback(ref, can): 
#     _, _, F1 = bert_scorer.score([can],ref) if type(ref) is type([]) else bert_scorer.score([can],[ref])

    _, _, F1 = bert_scorer.score([can],[ref])
    return F1.data[0].item()

def cider_score_callback(img_id, can_id, ref_id):
    return float(cider_score_map['{}-{}-{}'.format(img_id, can_id, ref_id)])

def rougeL_score_callback(ref, can):
    if type(ref) is type([]):
        refs = '\n'.join(ref)
        return rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=False).score(refs, can)
    else:
        return rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False).score(ref, can)

def rouge2_score_callback(ref, can):
    if type(ref) is not type([]):
        return rouge_scorer.RougeScorer(['rouge2'], use_stemmer=False).score(ref, can)
    else:
        # jackknifing
        combination_pool = itertools.combinations(ref, len(ref) - 1)
        rouge_scores = []
        for comb in combination_pool:
            comb_max = []
            for r in comb: 
                comb_max.append(rouge_scorer.RougeScorer(['rouge2'], use_stemmer=False).score(r, can)['rouge2'][2])
            rouge_scores.append(np.max(comb_max))
        return np.average(rouge_scores)
def rouge3_score_callback(ref, can):
    if type(ref) is not type([]):
        return rouge_scorer.RougeScorer(['rouge3'], use_stemmer=False).score(ref, can)
    else:
        # jackknifing
        combination_pool = itertools.combinations(ref, len(ref) - 1)
        rouge_scores = []
        for comb in combination_pool:
            comb_max = []
            for r in comb: 
                comb_max.append(rouge_scorer.RougeScorer(['rouge3'], use_stemmer=False).score(r, can)['rouge3'][2])
            rouge_scores.append(np.max(comb_max))
        return np.average(rouge_scores)

metrics_v3 = [
    {
        'metric_id': 'b_score',
        'callback': bleu_score,
#         'consensus_needed': True
    }, 
    {
        'metric_id': 'rouge2',
        'callback': rouge2_score_callback,
#         'consensus_needed': True
        'output_transform': (lambda x: x['rouge2'][2])
    },
    {
        'metric_id': 'rouge3',
        'callback': rouge3_score_callback,
#         'consensus_needed': True
        'output_transform': (lambda x: x['rouge3'][2])
    },
    {
        'metric_id': 'rougeL',
        'callback': rougeL_score_callback,
        'output_transform': (lambda x: x['rougeL'][2]),     # rougeLsum for multiple refs; rougeL for one  
#         'consensus_needed': True
    },
    {
        'metric_id': 'meteor',
        'callback': meteor_score,
#         'consensus_needed': True
    }, 
    {
        'metric_id': 'bert',
        'callback': bert_score_callback,
#         'consensus_needed': True                           # BERTScore can only be 1v1
    },
    {
        'metric_id': 'cider',
        'callback': cider_score_callback,
#         'consensus_needed': True
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
        perm_obj = {
            'scores': {
            }
        }
        
        # for cider and potentially similar metrics
#         og_councel = [c['caption'] for c in annotations_map[str(key)]]

        for permutation in captions_permutation_map[str(key)]:
            reference = permutation[0]['caption']
            candidate = permutation[1]['caption']
            candidate_id = permutation[1]['id']
            reference_id = permutation[0]['id']

            for metric_index in range(len(metric_callbacks)):
                
                    metric_id = metric_callbacks[metric_index]['metric_id']
                    metric_callback = metric_callbacks[metric_index]['callback']
                    out_transform = None if 'output_transform' not in metric_callbacks[metric_index] else metric_callbacks[metric_index]['output_transform']
                    if 'consensus_needed' in metric_callbacks[metric_index]:
#                         cand = {}
#                         cand[str(key)] = [candidate]
                        if metric_id == 'cider':
                            score = metric_callback(key, candidate_id)
                        else:
                            council = council_map['{}-{}'.format(key, candidate_id)]
                            score = metric_callback(council, candidate)
#                         print('CIDEr score: {}'.format(score))
#                         exit()
                    else:
                        if metric_id == 'cider':
                            score = metric_callback(key, candidate_id, reference_id)
                        else:
                            score = metric_callback(reference, candidate)
                    if out_transform != None:
#                         print('score: {}'.format(score))
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
total_list = loadMetadata('image_list_val.json',annotationsMeta['images'],make_image_list, False)
# train_image_list, test_image_list = splitImageList(total_list, 80)
annotations_map = loadMetadata('annotations_map_val.json', annotationsMeta['annotations'], make_annotations_map, False)
captions_permutation_map = loadMetadata('captions_permutation_map_val_v6.json', (total_list, annotations_map), make_captions_permutations, False)
scores = loadMetadata('scores_val_1v1_v6.json', (captions_permutation_map, metrics_v3, 'scores_val_1v1_v6.json', annotations_map), calculate_scores, False)
total_scores = loadMetadata('total_scores_val_1v1_v6.json', (total_list, scores, 'total_scores_val_1v1_v6.json'), filterTrainingScores, False)