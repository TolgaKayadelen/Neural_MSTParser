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
            # (TODO): decoder should return both predicted_heads and arc scores.
            predicted_heads, head_scores = decoder.Decode()
            predicted_heads[0] = -1
            #print("scores after decoding : {}".format(scores))
            # (TODO): asser length scores same as length tokens.
            assert len(predicted_heads) == len(sentence.token), "Number of tokens and heads must match!"
            assert len(head_scores) == len(sentence.token), "Number of tokens and scores must match!"
            # zip the token and its predicted head for each token.
            # TODO: zip also scores and sentence.token
            head_token = zip(predicted_heads, sentence.token)
            score_token = zip(head_scores, sentence.token)

            for i, token in enumerate(sentence.token):
                if token.word == "ROOT" or token.index == 0:
                    continue
                # necessary in cases where we evaluate on train data.
                token.ClearField("candidate_head")
                token.ClearField("selected_head")
                # insert the selected head into the token.
                # TODO: insert also the score to the token.
                token.selected_head.address=head_token[i][0]
                assert token.word == head_token[i][1].word, "Potential token mismatching!!"
                token.selected_head.arc_score = score_token[i][0]
                assert token.word == score_token[i][1].word, "Potential token mismatching!!"

            print(text_format.MessageToString(sentence, as_utf8=True))
            #heads = [token.selected_head.address for token in sentence.token]
            logging.info("DONE!")
            #print("predicted heads {}".format(predicted_heads))
            return sentence, predicted_heads
        else:
            raise Exception("Only mst decoding is available at this moment!")
