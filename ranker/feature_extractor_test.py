# -*- coding: utf-8 -*-

"""Unit tests for feature_extractor module"""

import unittest

from data.treebank import sentence_pb2
from google.protobuf import text_format

from ranker import feature_pb2
from ranker.feature_extractor import FeatureExtractor

from util import reader
from util import common

class FeatureExtractorTest(unittest.TestCase):
    """Tests for feature extractor"""

    def setUp(self):
        # Initialize a base document and a feature set for testing
        self.test_treebank_tr = reader.ReadTreebankTextProto("./data/testdata/features/kerem.pbtxt")
        self.test_sentence_tr = self.test_treebank_tr.sentence[0]
        self.test_sentence_en = reader.ReadSentenceTextProto("./data/testdata/generic/john_saw_mary.pbtxt")
        self.extractor = FeatureExtractor()

    def test_get_features(self):
        print("Running test_get_feetures..")
        token = self.test_sentence_tr.token[5]
        features = self.extractor.get_features(token, self.test_sentence_tr, n_prev=-2, n_next=2)
        expected_features = text_format.Parse("""
                                        word: "ettiği"
                                        pos: "Verb"
                                        category: "VERB"
                                        lemma: "et"
                                        morphology {
                                            name: "aspect"
                                            value: "perf"
                                        }
                                        morphology {
                                            name: "case"
                                            value: "nom"
                                        }
                                        morphology {
                                            name: "mood"
                                            value: "ind"
                                        }
                                        morphology {
                                            name: "number[psor]"
                                            value: "sing"
                                        }
                                        morphology {
                                            name: "person[psor]"
                                            value: "3"
                                        }
                                        morphology {
                                            name: "polarity"
                                            value: "pos"
                                        }
                                        morphology {
                                            name: "tense"
                                            value: "past"
                                        }
                                        morphology {
                                            name: "verbform"
                                            value: "part"
                                        }
                                        label: "compound:lvc"
                                        previous_token {
                                            word: "teslim"
                                            pos: "Noun"
                                            category: "NOUN"
                                            lemma: "teslim"
                                            morphology {
                                                name: "case"
                                                value: "nom"
                                            }
                                            morphology {
                                                name: "number"
                                                value: "sing"
                                            }
                                            morphology {
                                                name: "person"
                                                value: "3"
                                            }
                                            label: "obl"
                                            distance: -1
                                        }
                                        previous_token {
                                            word: "özgürlüğünü"
                                            pos: "Noun"
                                            category: "NOUN"
                                            lemma: "özgürlük"
                                            morphology {
                                                name: "case"
                                                value: "acc"
                                            }
                                            morphology {
                                                name: "number"
                                                value: "sing"
                                            }
                                            morphology {
                                                name: "number[psor]"
                                                value: "sing"
                                            }
                                            morphology {
                                                name: "person"
                                                value: "3"
                                            }
                                            morphology {
                                                name: "person[psor]"
                                                value: "3"
                                            }
                                            label: "obj"
                                            distance: -2
                                        }
                                        next_token {
                                            word: "için"
                                            pos: "PCNom"
                                            category: "ADP"
                                            lemma: "için"
                                            label: "case"
                                            distance: 1
                                        }
                                        next_token {
                                            word: "sanki"
                                            pos: "Adverb"
                                            category: "ADV"
                                            lemma: "sanki"
                                            label: "advmod"
                                            distance: 2
                                        }""", feature_pb2.Feature())
        self.assertEqual(features, expected_features)
        print("Passed!")



if __name__ == "__main__":
  unittest.main()
