# -*- coding: utf-8 -*-

"""Unit tests for max_span_tree module"""

import os
import unittest

from data.treebank import sentence_pb2
from google.protobuf import text_format
from mst import max_span_tree

_TESTDATA_DIR = "data/testdata"
_CYCLE_DIR = os.path.join(_TESTDATA_DIR, "cycles")
_CONNECTION_DIR = os.path.join(_TESTDATA_DIR, "connections")
_MST_DIR = os.path.join(_TESTDATA_DIR, "msts")
_TOKEN_DIR = os.path.join(_TESTDATA_DIR, "tokens")
_GENERIC_DIR = os.path.join(_TESTDATA_DIR, "generic")


def _read_file(path):
    with open(path, "r") as f:
        read = f.read()
    return read

def _read_connection_test_sentence(basename):
    path = os.path.join(_CONNECTION_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())

def _read_cycle_test_sentence(basename):
    path = os.path.join(_CYCLE_DIR, "{}.pbtxt".format(basename))
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

class MaximumSpanningTreeTest(unittest.TestCase):
    
    def test_GreedyMst(self):
        sentence = _read_mst_test_sentence("cyclic_no_selected_head")
        expected_mst_sentence = max_span_tree._GreedyMst(sentence)
        cyclic, path = max_span_tree._Cycle(expected_mst_sentence)
        self.assertTrue(cyclic)
        cyclic_mst = _read_mst_test_sentence("cyclic_mst")
        self.assertEqual(expected_mst_sentence, cyclic_mst)
    
    def test_Cycle(self):
        # Test sentences that have a cycle.
        cyclic_1 = _read_cycle_test_sentence("cyclic_sentence_1")
        cyclic, path = max_span_tree._Cycle(cyclic_1)
        self.assertTrue(cyclic)
        self.assertListEqual(path, [1,2,1])
        
        cyclic_2 = _read_cycle_test_sentence("cyclic_sentence_2")
        cyclic, path = max_span_tree._Cycle(cyclic_2)
        self.assertTrue(cyclic)
        self.assertListEqual(path, [1, 2, 3, 1])
        
        cyclic_3 = _read_cycle_test_sentence("cyclic_sentence_3")
        cyclic, path = max_span_tree._Cycle(cyclic_3)
        self.assertTrue(cyclic)
        self.assertListEqual(path, [2,3,4,2])
        
        cyclic_4 = _read_cycle_test_sentence("cyclic_sentence_4")
        cyclic, path = max_span_tree._Cycle(cyclic_4)
        self.assertTrue(cyclic)
        self.assertListEqual(path, [1,3,1])     
        
        cyclic_5 = _read_cycle_test_sentence("cyclic_sentence_5")
        cyclic, path = max_span_tree._Cycle(cyclic_5)
        self.assertTrue(cyclic)
        self.assertListEqual(path, [2,3,2])
        
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
    
    def test_ConnectSentenceNodes(self):
        non_connected = _read_connection_test_sentence("non_connected_sentence")
        connected = _read_connection_test_sentence("connected_sentence")
        test_connected = max_span_tree.ConnectSentenceNodes(non_connected)
        #print(text_format.MessageToString(test_connected, as_utf8=True))
        self.assertEqual(connected, test_connected)
    
    def test_DropCandidateHeads(self):
        no_ch = _read_connection_test_sentence("non_connected_sentence")
        ch = _read_connection_test_sentence("connected_sentence")
        max_span_tree._DropCandidateHeads(ch)
        self.assertEqual(no_ch, ch)
    
    
    def test_Contract(self):
        cyclic = _read_cycle_test_sentence("cyclic_sentence_1")
        _, cycle_path = max_span_tree._Cycle(cyclic)
        _, original_edges, contracted = max_span_tree._Contract(cyclic, cycle_path)
        expected_sentence = _read_mst_test_sentence("cyclic_sentence_1_contracted")
        self.assertEqual(expected_sentence, contracted)
        expected_edges = {
            0: [(1, 9.0), (2, 10.0), (3, 9.0)],
            1: [(2, 20.0), (3, 3.0)],
            2: [(1, 30.0), (3, 30.0)],
            3: [(1, 11.0), (2, 0.0)]
        }
        self.assertEqual(expected_edges, original_edges)
        
    
    def test_GetCycleScore(self):
        cyclic = _read_cycle_test_sentence("cyclic_sentence_1")
        _, cycle_path = max_span_tree._Cycle(cyclic)
        self.assertListEqual(cycle_path, [1,2,1])
        cycle_tokens, cycle_score = max_span_tree._GetCycleScore(cyclic, cycle_path)
        self.assertEqual(cycle_tokens, cyclic.token[1:3])
        self.assertEqual(cycle_score, 50)
    
    
    def test_GetOriginalEdges(self):
        cyclic = _read_cycle_test_sentence("cyclic_sentence_1")
        original_edges = max_span_tree._GetOriginalEdges(cyclic)
        expected_edges = {
            0: [(1, 9.0), (2, 10.0), (3, 9.0)],
            1: [(2, 20.0), (3, 3.0)],
            2: [(1, 30.0), (3, 30.0)],
            3: [(1, 11.0), (2, 0.0)]
        }
        self.assertDictEqual(expected_edges, original_edges)
    
    def test_RedirectIncomingArcs(self):
        #(TODO): expand this test to more examples
        cyclic = _read_cycle_test_sentence("cyclic_sentence_1")
        cyclic.length = len(cyclic.token)
        cycle_tokens = cyclic.token[1:3] #tokens 1 and 2
        cycle_score = 50
        new_token_index = cyclic.token[-1].index + 1
        new_token = cyclic.token.add()
        new_token.word = "cycle_token"
        new_token.index = new_token_index
        cyclic.length += 1
        max_span_tree._RedirectIncomingArcs(cycle_tokens, new_token, cycle_score)
        expected_token = _read_test_token("cycle_1_incoming")
        self.assertEqual(expected_token, new_token)
    
    def test_RedirectOutgoingArcs(self):
        #(TODO): expand this test to more examples
        cyclic = _read_cycle_test_sentence("cyclic_sentence_1")
        cyclic.length = len(cyclic.token)
        cycle_tokens = cyclic.token[1:3] # tokens 1 and 2
        outcycle_tokens = [
            token for token in cyclic.token if not token in cycle_tokens
        ] # tokens 0 and 3
        #print(outcycle_tokens)    
        new_token_index = cyclic.token[-1].index + 1
        new_token = cyclic.token.add()
        new_token.index = new_token_index
        cyclic.length += 1
        max_span_tree._RedirectOutgoingArcs(cycle_tokens, outcycle_tokens, new_token)
        expected_token = _read_test_token("cycle_1_outgoing")
        changed_token = max_span_tree._GetTokenByAddressAlt(cyclic.token, 3)
        self.assertEqual(expected_token, changed_token)
    
    
    def test_DropCycleTokens(self):
        cyclic = _read_cycle_test_sentence("cyclic_sentence_1")
        cycle_tokens = cyclic.token[1:3]
        expected_sentence = max_span_tree._DropCycleTokens(cyclic, cycle_tokens)
        self.assertTrue(len(expected_sentence.token) == 2)
        self.assertTrue(expected_sentence.token[0].word == "None")
        self.assertTrue(expected_sentence.token[1].word == "Mary")
    
    def test_Reconstruct(self):
        cyclic = _read_cycle_test_sentence("cyclic_sentence_1")
        cycle_tokens, cycle_path = max_span_tree._Cycle(cyclic)
        #new_token, original_edges, contracted = max_span_tree._Contract(cyclic, cycle_path)
        new_sentence = max_span_tree.ChuLiuEdmonds(cyclic)
        #print(new_sentence)
        
    def test_GetTokenIndex(self):
        pass
    
    def test_GetTokenByAddress(self):
        pass
    
        
       
if __name__ == "__main__":
  unittest.main()