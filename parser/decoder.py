# -*- coding: utf-8 -*-

"""Dependency Tree Decoder."""

from mst.max_span_tree import ChuLiuEdmonds as cle
from util.common import DropDummyTokens

class Decoder:
    def __init__(self, decoding="mst"):
        assert decoding in ("mst", "eisner")
        self.decoding = decoding
        
    def __call__(self, sentence, scores):
        if self.decoding == "mst":
            sentence = DropDummyTokens(sentence)
            for token in sentence.token:
                if token.word == "ROOT" or token.index == 0:
                    continue
                token.ClearField("candidate_head")
                token.ClearField("selected_head")
                for i in range(scores.shape[1]):
                    token.candidate_head.add(
                        arc_score = scores[token.index, i],
                        address = i
                    )
            #print(sentence)
            #print(cle(sentence)), [token.selected_head.address for token in sentence.token]
            return cle(sentence), [token.selected_head.address for token in sentence.token]
        else:
            raise Exception("Only mst decoding is available at this moment!")
