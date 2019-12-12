# -*- coding: utf-8 -*-

"""Dependency Tree Decoder."""

from google.protobuf import text_format
from mst.max_span_tree_simple import MST as mst_decoder
from util import common
import numpy as np

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

class Decoder:
    def __init__(self, decoding="mst"):
        assert decoding in ("mst", "eisner")
        self.decoding = decoding

    def __call__(self, sentence, scores):
        #logging.info("Starting decoding the tree..")
        #logging.info("scores are: {}".format(scores))
        if self.decoding == "mst":
            sentence = common.DropDummyTokens(sentence)
            # heavily penalize self referential head selection.
            for i in range(scores.shape[0]):
                scores[i][i] = -100000.
            #print("scores at decode time: {}".format(scores))
            #raw_input("Press Yes to continue: ")
            decoder = mst_decoder(scores)
            predicted_heads, head_scores = decoder.Decode()
            predicted_heads[0] = -1
            #print("scores after decoding : {}".format(scores))
            assert len(predicted_heads) == len(sentence.token), "Number of tokens and heads must match!"
            assert len(head_scores) == len(sentence.token), "Number of tokens and scores must match!"
            # zip the token and its predicted head for each token.
            head_token = zip(predicted_heads, sentence.token)
            # zip the token and its head arc score for each token.
            score_token = zip(head_scores, sentence.token)

            for i, token in enumerate(sentence.token):
                if token.word == "ROOT" or token.index == 0:
                    continue
                # necessary in cases where we evaluate on train data.
                token.ClearField("candidate_head")
                token.ClearField("selected_head")
                # insert the selected head into the token.
                #logging.info("Inserting the decoded arcs and scores to the sentence.")
                #(TODO): we should also insert the sentence score.
                token.selected_head.address=head_token[i][0]
                assert token.word == head_token[i][1].word, "Potential token mismatching!!"
                token.selected_head.arc_score = score_token[i][0]
                assert token.word == score_token[i][1].word, "Potential token mismatching!!"
            # the total score of the sentence based on arc scores.
            sentence.score = common.GetSentenceWeight(sentence)
            #print(text_format.MessageToString(sentence, as_utf8=True))
            #heads = [token.selected_head.address for token in sentence.token]
            #logging.info("DONE!")
            #print("predicted heads {}".format(predicted_heads))
            return sentence, predicted_heads
        else:
            raise Exception("Only mst decoding is available at this moment!")
