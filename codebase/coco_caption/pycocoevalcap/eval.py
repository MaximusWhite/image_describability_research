__author__ = 'vrama91'

import os
import json

from collections import defaultdict
from .tokenizer.ptbtokenizer import PTBTokenizer
# from .bleu.bleu import Bleu
# from .meteor.meteor import Meteor
# from .rouge.rouge import Rouge
from .cider.cider import Cider

class COCOEvalCap:
    def __init__(self, pathToData, refName, candName, dfMode = "corpus"):
        """
        Reference file: list of dict('image_id': image_id, 'caption': caption).
        Candidate file: list of dict('image_id': image_id, 'caption': caption).
        :params: refName : name of the file containing references
        :params: candName: name of the file containing cnadidates
        """
        self.eval = {}
        self._refName = refName
        self._candName = candName
        self._pathToData = pathToData
        self._dfMode = dfMode

    def evaluate(self):
        """
        Load the sentences from json files
        """
        def readJson(refName, candName):

            path_to_ref_file = os.path.join(self._pathToData, refName)
            path_to_cand_file = os.path.join(self._pathToData, candName)

            ref_list = json.loads(open(path_to_ref_file, 'r').read())
            cand_list = json.loads(open(path_to_cand_file, 'r').read())

            gts = defaultdict(list)
            res = defaultdict(list)
            # change of naming convention from ref to gts
            for l in ref_list:
                gts[l['image_id']].append({"caption": l['caption']})

            # change of naming convention from cand to res
            for l in cand_list:
                res[l['image_id']].append({"caption": l['caption']})

            return gts, res

        print('Loading Data...')
        gts, res = readJson(self._refName, self._candName)
        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        
        
        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
#             (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#             (Meteor(),"METEOR"),
#             (Rouge(), "ROUGE_L"),
            (Cider(self._dfMode), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
            else:
                self.setEval(score, method)

    def setEval(self, score, method):
        self.eval[method] = score

