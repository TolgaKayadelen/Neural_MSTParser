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
        
        #cyclic_list = [cyclic_1, cyclic_2, cyclic_3, cyclic_4, cyclic_5, cyclic_5, cyclic_6]
        cyclic_list = [cyclic_2]
        
        for i, sentence in enumerate(cyclic_list):
            # Create the numpy array from candidate heads
            if not sentence.HasField("length"):
                sentence.length = len(sentence.token)
            scores = np.zeros((sentence.length, sentence.length))
            for token in sentence.token:
                scores[token.index][token.index] = -1.
                for ch in token.candidate_head:
                    scores[token.index][ch.address] = ch.arc_score
            print("Scores in the test function: {}".format(scores))
        
        decoder = max_span_tree_simple.MST(scores)
        vertices = set((0,))
        edges = defaultdict(set)
        for dep, head in enumerate(decoder.heads[decoder.tokens]):
            vertices.add(dep+1)
            edges[head].add(dep+1)
        print("vertices: {}".format(vertices))
        print("edges: {}".format(edges))
        cycle = decoder._GetCycle(vertices, edges)
        #self.assertTrue(cycle, [1,2])
        print("cycle: {}".format(cycle))
        
                    
                    
        
        
        
        '''
        cyclic_2 = _read_cycle_test_sentence("cyclic_sentence_2")
        cyclic, path = max_span_tree._Cycle(cyclic_2)
        self.assertTrue(cyclic)
        self.assertListEqual(path, [1, 2, 3])
        
        
        cyclic, path = max_span_tree._Cycle(cyclic_3)
        self.assertTrue(cyclic)
        self.assertListEqual(path, [2,3,4])
        
        
        cyclic, path = max_span_tree._Cycle(cyclic_4)
        self.assertTrue(cyclic)
        self.assertListEqual(path, [1,3])     
        
        
        cyclic, path = max_span_tree._Cycle(cyclic_5)
        self.assertTrue(cyclic)
        self.assertListEqual(path, [2,3])
        
        
        cyclic, path = max_span_tree._Cycle(cyclic_6)
        self.assertTrue(cyclic)
        self.assertListEqual(path, [2,3,4])
        
        # Test sentences that don't have a cycle
        noncyclic_1 = _read_cycle_test_sentence("noncyclic_sentence_1")
        cyclic, path = max_span_tree._Cycle(noncyclic_1)
        self.assertFalse(cyclic)
        
        noncyclic_2 = _read_cycle_test_sentence("noncyclic_sentence_2")
        cyclic, path = max_span_tree._Cycle(noncyclic_2)
        self.assertFalse(cyclic)
        
        noncyclic_3 = _read_cycle_test_sentence("noncyclic_sentence_3")
        cyclic, path = max_span_tree._Cycle(noncyclic_3)
        self.assertFalse(cyclic)
        logging.info("Passed!")
        '''
  
       
if __name__ == "__main__":
  unittest.main()