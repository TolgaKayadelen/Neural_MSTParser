# -*- coding: utf-8 -*-

"""Unit tests for feature_extractor module"""

import unittest

from data.treebank import sentence_pb2
from google.protobuf import text_format
from learner.feature_extractor import FeatureExtractor
from util import reader

class FeatureExtractorTest(unittest.TestCase):
    """Tests for feature extractor"""
    
    def setUp(self):
        # Initialize a base document and a feature set for testing
        self.test_sentence = reader.ReadSentenceProto("./data/treebank/sentence_4.protobuf")
        self.feature_file = "./learner/features.txt"
    
    def test_GetFeatures(self):
        head = self.test_sentence.token[3]
        child = self.test_sentence.token[1]
        # Initialize the extractor with a feature file
        extractor = FeatureExtractor(self.feature_file)
        function_features = extractor.GetFeatures(self.test_sentence, head, child, use_tree_features=True)
        expected_features = {
            'head_0_word + head_0_pos': 'özgürlüğünü_Noun',
            'head_0_word': 'özgürlüğünü',
            'head_0_pos': 'Noun',
            'child_0_word + child_0_pos': 'Kerem_Prop',
            'child_0_word': 'Kerem',
            'child_0_pos': 'Prop',
            'head_0_word + head_0_pos + child_0_word + child_0_pos': 'özgürlüğünü_Noun_Kerem_Prop',
            'head_0_pos + child_0_word + child_0_pos': 'Noun_Kerem_Prop',
            'head_0_word + child_0_word + child_0_pos': 'özgürlüğünü_Kerem_Prop',
            'head_0_word + head_0_pos + child_0_pos': 'özgürlüğünü_Noun_Prop',
            'head_0_word + head_0_pos + child_0_word': 'özgürlüğünü_Noun_Kerem',
            'head_0_word + child_0_word': 'özgürlüğünü_Kerem',
            'head_0_pos + child_0_pos': 'Noun_Prop',
            'head_0_pos + between_0_pos + child_0_pos': 'Noun_Punc_Prop',
            'head_0_pos + head_1_pos + child_-1_pos + child_0_pos': 'Noun_Noun_ROOT_Prop',
            'head_-1_pos + head_0_pos + child_-1_pos + child_0_pos': 'Punc_Noun_ROOT_Prop',
            'head_0_pos + head_1_pos + child_0_pos + child_1_pos': 'Noun_Noun_Prop_Punc',
            'head_-1_pos + head_0_pos + child_0_pos + child_1_pos': 'Punc_Noun_Prop_Punc',
            'head_0_word + head_0_up_word': 'özgürlüğünü_teslim',
            'head_0_pos + head_0_up_pos': 'Noun_Noun',
            'head_0_word + head_0_down_word': 'özgürlüğünü_None',
            'head_0_pos + head_0_down_pos': 'Noun_None',
            'child_0_word + child_0_up_word': 'Kerem_rahatlamıştı',
            'child_0_pos + child_0_up_pos': 'Prop_Verb',
            'child_0_word + child_0_down_word': 'Kerem_None',
            'child_0_pos + child_0_down_pos': 'Prop_None'
        }
        self.assertItemsEqual(function_features, expected_features)
        
    
if __name__ == "__main__":
  unittest.main()