import copy
import sys

# Use cPickle if it is installed, as it is much faster
if 'cPickle' in sys.modules:
    import cPickle as pickle
else:
    import pickle

from collections import defaultdict
import numpy as np
import math
import os
from pathlib import Path
import pdb
import json


class CiderChad(object):
    def copy(self):
        new = CiderScorer(n=self.n)
        new.ccands = copy.copy(self.ccands)
        new.cgts = copy.copy(self.cgts)
        return new

    def __init__(self, cands=None, gts=None, n=4, sigma=6.0, df_config=None):
        self.n = n
        self.sigma = sigma
        self.cgts = []
        self.ccands = []
        self.img_ids = []
        self.load_df(df_config)
    
    def load_df(self, config=None):
        # config = [{filename to load from}, {list of gts objects for df}]
        if config is None: 
            print('Give me at least something for df...')
            exit()
        if config[0] != None:
            self.document_frequency = json.load(open(config[0], 'r'))
            print('df loaded')
        else:
             print('cool')
#             words = s.split()
#             counts = defaultdict(int)
#             for k in range(1,n+1):
#                 for i in range(len(words)-k+1):
#                     ngram = tuple(words[i:i+k])
#                     counts[ngram] += 1
#             return counts
        
    def test(self, s, n=4):
        words = s.split()
        counts = defaultdict(int)
        for k in range(1,n+1):
            for i in range(len(words)-k+1):
                ngram = tuple(words[i:i+k])
                counts[ngram] += 1
        return counts
        