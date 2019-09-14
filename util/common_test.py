# -*- coding: utf-8 -*-

"""Unit tests for max_span_tree module"""

import os
import unittest

from data.treebank import sentence_pb2
from google.protobuf import text_format
from util import common

_TESTDATA_DIR = "data/testdata"
_COMMON_DIR = os.path.join(_TESTDATA_DIR, "common")
_CONNECTION_DIR = os.path.join(_TESTDATA_DIR, "connections")

def _read_file(path):
    with open(path, "r") as f:
        read = f.read()
    return read

def _read_common_test_sentence(basename):
    path = os.path.join(_COMMON_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())

def _read_connection_test_sentence(basename):
    path = os.path.join(_CONNECTION_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())

class CommonTest(unittest.TestCase):

    def setUp(self):
        self._john_saw_mary = _read_common_test_sentence("john_saw_mary")
        self._kerem = _read_common_test_sentence("kerem")

    def test_ExtendSentence(self):
        print("Running test_ExtendSentence")
        function_extended_sentence = common.ExtendSentence(self._john_saw_mary)
        expected_extended_sentence = _read_common_test_sentence("john_saw_mary_extended")
        self.assertEqual(function_extended_sentence, expected_extended_sentence)
        print("Passed!")

    def test_GetRightMostChild(self):
        print("Running test_GetRightMostChild")
        # main verb = rahatlamisti
        rm_main_verb = common.GetRightMostChild(self._kerem, self._kerem.token[8])
        # light verb = teslim
        rm_light_verb = common.GetRightMostChild(self._kerem, self._kerem.token[4])
        self.assertTrue(rm_main_verb.word == ".")
        #print(rm_light_verb.word)
        self.assertTrue(rm_light_verb.word == u"için")
        print("Passed!")

    def test_GetBetweenTokens(self):
        print("Running test_GetBetweenTokens")
        expected_tokens = [[u"John", u"saw"], [None], [None], [u"John",u"saw"]]
        start = 0
        end = 3
        for i in range(end+1):
            head = self._john_saw_mary.token[start]
            child = self._john_saw_mary.token[end]
            between_tokens = common.GetBetweenTokens(self._john_saw_mary, head, child, dummy=0)
            start += 1
            end -= 1
            self.assertTrue(expected_tokens[i] == list(token.word if token else None for token in between_tokens))

        # check the case where the sentence gets extended.
        head = self._john_saw_mary.token[0] # Root
        child = self._john_saw_mary.token[2] # saw
        expected_between_tokens = [u"John"]
        extended_sentence = common.ExtendSentence(self._john_saw_mary)
        between_tokens = common.GetBetweenTokens(self._john_saw_mary, head, child, dummy=1)
        self.assertEqual(expected_between_tokens, list(token.word for token in between_tokens))
        print("Passed!")

    def test_GetValue(self):
        print("Running test_GetValue")
        tokens = self._john_saw_mary.token
        expected_categories = ["ROOT", "Noun", "Verb", "Noun"]
        expected_lemmas = ["ROOT", u"John", u"see", u"Mary"]
        function_categories = [common.GetValue(token, "pos") for token in tokens]
        function_lemmas = [common.GetValue(token, "lemma") for token in tokens]
        self.assertEqual(expected_lemmas, function_lemmas)
        self.assertEqual(expected_categories, function_categories)
        print("Passed!")

    def test_ConnectSentenceNodes(self):
        print("Running test_ConnectSentenceNodes")
        non_connected = _read_connection_test_sentence("non_connected_sentence")
        connected = _read_connection_test_sentence("connected_sentence")
        test_connected = common.ConnectSentenceNodes(non_connected)
        #print(text_format.MessageToString(test_connected, as_utf8=True))
        self.assertEqual(connected, test_connected)
        print("Passed!")

    def test_GetTokenByAddress(self):
        print("Running test GetTokenByAddress..")
        test_sentence = _read_common_test_sentence("john_saw_mary")
        tokens = list(test_sentence.token)
        john = tokens[1]
        search = john.selected_head.address #saw
        function_head = common.GetTokenByAddress(tokens, search)
        expected_head = text_format.Parse("""
            word: "saw"
            lemma: "see"
            category: "VERB"
            pos: "Verb"
            candidate_head {
                address: 0
                arc_score: 10.0
            }
            candidate_head {
                address: 1
                arc_score: 5.0
            }
            candidate_head {
                address: 3
                arc_score: 0.0
            }
            selected_head {
                address: 0
                arc_score: 10.0
            }
            index: 2
            """, sentence_pb2.Token())
        self.assertEqual(function_head, expected_head)

        # Test with an extendend sentence.
        test_sentence_ext = _read_common_test_sentence("john_saw_mary_extended")
        words = [u'ROOT', u'John', u'saw', u'Mary']
        for i in range(4):
            found = common.GetTokenByAddress(tokens, i)
            self.assertEqual(found.word, words[i])
        print("Passed!")

    def test_DropDummyTokens(self):
        print("Running test DropDummyTokens..")
        test_sentence = _read_common_test_sentence("john_saw_mary_extended")
        function_sentence = common.DropDummyTokens(test_sentence)
        expected_sentence = _read_common_test_sentence("john_saw_mary")
        self.assertEqual(function_sentence, expected_sentence)
        print("Passed!")

    def testGetSentenceWeight(self):
        print("Running testGetSentenceWeight..")
        test1 = _read_common_test_sentence("john_saw_mary_extended")
        self.assertEqual(common.GetSentenceWeight(test1), 70.0)
        test2 = _read_common_test_sentence("john_saw_mary")
        self.assertEqual(common.GetSentenceWeight(test2), 70.0)

        print("Passed!")


if __name__ == "__main__":
  unittest.main()
