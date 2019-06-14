# -*- coding: utf-8 -*-

"""Dependency Tree Decoder."""

from google.protobuf import text_format
from mst.max_span_tree import ChuLiuEdmonds as cle
from util.common import PPrintTextProto
from util.common import DropDummyTokens
from mst.max_span_tree_simple import MST as mst_decoder
from sys import maxint
import numpy as np

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

class Decoder:
    def __init__(self, decoding="mst"):
        assert decoding in ("mst", "eisner")
        self.decoding = decoding
        
    def __call__(self, sentence, scores):
        logging.info("Starting decoding the tree..")
        if self.decoding == "mst":
            sentence = DropDummyTokens(sentence)
            # heavily penalize self referential head selection.
            for i in range(scores.shape[0]):
                scores[i][i] = -100000.
            #print("scores at decode time: {}".format(scores))
            #raw_input("Press Yes to continue: ")
            decoder = mst_decoder(scores)
            predicted_heads = decoder.Decode()
            predicted_heads[0] = -1
            #print("scores after decoding : {}".format(scores))
            #raw_input("Press Yes to continue: ")
            assert len(predicted_heads == len(sentence.token)), "Number of tokens and heads must match!"
            head_token = zip(predicted_heads, sentence.token)
            
            for i, token in enumerate(sentence.token):
                if token.word == "ROOT" or token.index == 0:
                    continue
                # necessary in cases where we evaluate on train data.
                token.ClearField("candidate_head")
                token.ClearField("selected_head")
                # insert the selected head into the token.
                token.selected_head.address=head_token[i][0]
                assert token.word == head_token[i][1].word, "Potential token mismatching!!"
            
            # remove later
            if sentence.token[1].word == "Kerem":
                print("Predicted Sentence: ")
                PPrintTextProto(sentence)
                #for i in range(scores.shape[1]):
                #    if token.index == i:
                        #print(token.index, i)
                #        continue
                #    token.candidate_head.add(
                #        arc_score = scores[token.index, i],
                #        address = i
                #    )
            
            #print(text_format.MessageToString(sentence, as_utf8=True))
            
            #heads = [token.selected_head.address for token in sentence.token]
            #print(text_format.MessageToString(cle_sentence, as_utf8=True))
            logging.info("DONE!")
            #print("predicted heads {}".format(predicted_heads))
            return sentence, predicted_heads
        else:
            raise Exception("Only mst decoding is available at this moment!")
        
