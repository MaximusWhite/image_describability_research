# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from .cider_scorer import CiderScorer
import pdb
from pathlib import Path
import os
import sys
import numpy as np

# Use cPickle if it is installed, as it is much faster
if 'cPickle' in sys.modules:
    import cPickle as pickle
else:
    import pickle


class Cider:
    """
    Main Class to compute the CIDEr metric

    """
    def get_df(self):
        return self.cider_scorer.get_df()
    
    def __init__(self, df, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        self._dfMode = df
        self.cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
        #### move doc frequency load outside of scorer to make things faster
        p = Path(__file__).parents[2]
        #### HAVE TO MODIFY TO UNPICKLE STUFF PICKLED WITH PYTHON 2
        if df != 'custom' and df != 'corpus':
            f = open(os.path.join(p, 'data', df + '.p'), 'rb')
#             docFreq = pickle.load(f, encoding="latin1")
            docFreq = pickle.load(f)

            self._docFreq = docFreq['df']
            self._ref_len = np.log(float(docFreq['ref_len']))
        else: 
            self._docFreq = None
            self._ref_len = None
    
    def prepare_df(self, gts):
        print('preparing df...')
#         assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        for id in imgIds:
#             hypo = res[id]
            ref = gts[id]
            
#             print('hypo {}: {}'.format(id, hypo))
#             print('ref {}: {}'.format(id, ref))
            
            # Sanity check.
#             assert(type(hypo) is list)
#             assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            self.cider_scorer.add_for_df((None, ref))
        self.cider_scorer.compute_doc_freq()
        self.cider_scorer.refresh_sets()
        print('df ready...')
    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """
        
#         if self._dfMode == 'custom':
#             self.prepare_df(gts, res)
        
#         print(len(res.keys()))
#         print(len(gts.keys()))
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

#         cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
        
        print('adding data...', end="")
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            
#             print('hypo {}: {}'.format(id, hypo))
#             print('ref {}: {}'.format(id, ref))
            
            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            self.cider_scorer += (hypo[0], ref, id)
        print('done')
        
        (score, scores) = self.cider_scorer.compute_score(self._dfMode, self._docFreq, self._ref_len)

        return score, scores

    def method(self):
        return "CIDEr"
