# -*- coding: utf-8 -*-

"""Unit tests for max_span_tree module"""

import os
import unittest


import numpy as np
from mst import max_span_tree_simple
from collections import defaultdict
from data.treebank import sentence_pb2
from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

_TESTDATA_DIR = "data/testdata"
_CYCLE_DIR = os.path.join(_TESTDATA_DIR, "cycles")
_CONNECTION_DIR = os.path.join(_TESTDATA_DIR, "connections")
_MST_DIR = os.path.join(_TESTDATA_DIR, "msts")
_TOKEN_DIR = os.path.join(_TESTDATA_DIR, "tokens")
_GENERIC_DIR = os.path.join(_TESTDATA_DIR, "generic")
_CHULIUEDMONDS_DIR = os.path.join(_TESTDATA_DIR, "chuliuedmonds")


def _read_file(path):
    with open(path, "r") as f:
        read = f.read()
    return read

def _read_cycle_test_sentence(basename):
    path = os.path.join(_CYCLE_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())

def _read_connection_test_sentence(basename):
    path = os.path.join(_CONNECTION_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())

def _read_mst_test_sentence(basename):
    path = os.path.join(_MST_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())

def _read_generic_test_sentence(basename):
    path = os.path.join(_GENERIC_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())

def _read_test_token(basename):
    path = os.path.join(_TOKEN_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Token())

def _read_chuliuedmonds_test_sentence(basename):
    path = os.path.join(_CHULIUEDMONDS_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())

class MaximumSpanningTreeTest(unittest.TestCase):
    
      
    def test_GetCycle(self):
        logging.info("Running CycleTest..")
        # Test sentences that have a cycle.
        
        cyclic_1 = _read_cycle_test_sentence("cyclic_sentence_1")
        cyclic_2 = _read_cycle_test_sentence("cyclic_sentence_2")
        cyclic_3 = _read_cycle_test_sentence("cyclic_sentence_3")
        cyclic_4 = _read_cycle_test_sentence("cyclic_sentence_4")
        cyclic_5 = _read_cycle_test_sentence("cyclic_sentence_5")
        cyclic_6 = _read_cycle_test_sentence("cyclic_sentence_6")
        cyclic_7 = _read_cycle_test_sentence("cyclic_sentence_7")
        
        cyclic_list = [cyclic_1, cyclic_2, cyclic_6, cyclic_7]
        expected_cycles = [[1,2], [1,2,3], [2,3,4], [1,2]]
        
        for i, sentence in enumerate(cyclic_list):
            # Create the numpy array from candidate heads
            if not sentence.HasField("length"):
                sentence.length = len(sentence.token)
            scores = np.zeros((sentence.length, sentence.length))
            for token in sentence.token:
                scores[token.index][token.index] = -1.
                for ch in token.candidate_head:
                    scores[token.index][ch.address] = ch.arc_score
        
            decoder = max_span_tree_simple.MST(scores)
            vertices = set((0,))
            edges = defaultdict(set)
            for dep, head in enumerate(decoder.heads[decoder.tokens]):
                vertices.add(dep+1)
                edges[head].add(dep+1)
            #print("vertices: {}".format(vertices))
            #print("edges: {}".format(edges))
            cycle = decoder._GetCycle(vertices, edges)
            self.assertEqual(list(cycle[0]), expected_cycles[i])
            #print("cycle: {}".format(cycle))
            #print("-------------------------------------------")
        
        print("Passed!")
    
    def testDecode(self):
        
        sentence_1 = _read_chuliuedmonds_test_sentence("cyclic_sentence_1")
        if not sentence_1.HasField("length"):
            sentence_1.length = len(sentence_1.token)
        scores = np.zeros((sentence_1.length, sentence_1.length))
        for token in sentence_1.token:
            scores[token.index][token.index] = -1.
            for ch in token.candidate_head:
                scores[token.index][ch.address] = ch.arc_score
    
        decoder = max_span_tree_simple.MST(scores)
        heads = decoder.Decode()
        self.assertListEqual(list(heads), [0,2,0,2])
        
        
        #print("------------------------------------------------")
        sentence_2 = _read_chuliuedmonds_test_sentence("cyclic_sentence_2")
        if not sentence_2.HasField("length"):
            sentence_2.length = len(sentence_2.token)
        scores = np.zeros((sentence_2.length, sentence_2.length))
        for token in sentence_2.token:
            scores[token.index][token.index] = -1.
            for ch in token.candidate_head:
                scores[token.index][ch.address] = ch.arc_score
    
        decoder = max_span_tree_simple.MST(scores)
        heads = decoder.Decode()
        self.assertEqual(list(heads), [0, 2, 0, 4, 2])

        
        
        #print("------------------------------------------------")
        sentence_5 = _read_chuliuedmonds_test_sentence("cyclic_sentence_5")
        if not sentence_5.HasField("length"):
            sentence_5.length = len(sentence_5.token)
        scores = np.zeros((sentence_5.length, sentence_5.length))
        for token in sentence_5.token:
            scores[token.index][token.index] = -1.
            for ch in token.candidate_head:
                scores[token.index][ch.address] = ch.arc_score
    
        decoder = max_span_tree_simple.MST(scores)
        heads = decoder.Decode()
        self.assertEqual(list(heads), [0,2,3,4,0])

        
        #print("------------------------------------------------")
        #Test on a longer dummy sentence.
        w2i = defaultdict(lambda: len(w2i))
        sentence_dummy = np.arange(10)
        sentence_ids = sentence_dummy
        num_words = len(sentence_dummy)
        i2w = {i: w for w, i in w2i.items()}

        #print(sentence_dummy)
        #print(sentence_ids)
        #print(num_words)
        
        np.random.seed(seed=42)
        scores = np.array(
            
            [[ 37.45401188,  95.07143064,  73.19939418,  59.86584842,  15.60186404,
               15.59945203,   5.80836122,  86.61761458,  60.11150117,  70.80725778],
             [  2.05844943,  96.99098522,  83.24426408,  21.23391107,  18.18249672,
               18.34045099,  30.4242243 ,  52.47564316,  43.19450186,  29.12291402],
             [ 61.18528947,  13.94938607,  29.21446485,  36.63618433,  45.60699842,
               78.51759614,  19.96737822,  51.42344384,  59.24145689,   4.64504127],
             [ 60.75448519,  17.05241237,   6.5051593 ,  94.88855373,  96.56320331,
               80.83973481,  30.46137692,   9.7672114 ,  68.42330265,  44.01524937],
             [ 12.20382348,  49.51769101,   3.43885211,  90.93204021,  25.87799816,
               66.25222844,  31.17110761,  52.00680212,  54.67102793,  18.48544555],
             [ 96.95846278,  77.51328234,  93.94989416,  89.48273504,  59.78999788,
               92.1874235 ,   8.84925021,  19.59828624,   4.52272889,  32.53303308],
             [ 38.86772897,  27.13490318,  82.87375092,  35.67533267,  28.09345097,
               54.26960832,  14.0924225 ,  80.21969808,   7.45506437,  98.68869366],
             [ 77.22447693,  19.87156815,   0.55221171,  81.54614285,  70.68573438,
               72.9007168 ,  77.12703467,   7.40446517,  35.84657285,  11.58690595],
             [ 86.31034259,  62.32981268,  33.08980249,   6.35583503,  31.09823217,
               32.5183322 ,  72.96061783,  63.75574714,  88.72127426,  47.22149252],
             [ 11.95942459,  71.32447872,  76.07850486,  56.12771976,  77.096718,
               49.37955964,  52.27328294,  42.75410184,   2.54191267,  10.7891427 ]]
            
        )
        
        length = scores.shape[0]
        scores = scores * (1 - np.eye(length))
        
        decoder = max_span_tree_simple.MST(scores.T)
        heads = decoder.Decode()
        self.assertEqual(list(heads), [0,0,5,7,3,3,7,1,3,6])
                    
  
       
if __name__ == "__main__":
  unittest.main()