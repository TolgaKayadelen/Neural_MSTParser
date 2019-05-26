# -*- coding: utf-8 -*-

"""Unit tests for feature_extractor module"""

import unittest

from data.treebank import sentence_pb2
from google.protobuf import text_format
from learner.feature_extractor import FeatureExtractor
from util import reader
from util import common

class FeatureExtractorTest(unittest.TestCase):
    """Tests for feature extractor"""
    
    def setUp(self):
        # Initialize a base document and a feature set for testing
        self.test_sentence_tr = reader.ReadSentenceProto("./data/treebank/sentence_4.protobuf")
        self.test_sentence_en = reader.ReadSentenceTextProto("./data/testdata/generic/john_saw_mary.pbtxt")
        self.extractor = FeatureExtractor()
    
    def test_GetFeatures(self):
        head = self.test_sentence_tr.token[3]
        child = self.test_sentence_tr.token[1]
        # Initialize the extractor with a feature file
        function_features = self.extractor.GetFeatures(self.test_sentence_tr, head, child, use_tree_features=True)
        #print(text_format.MessageToString(function_features, as_utf8=True))
        features_dict = dict((feature.name, feature.value) for feature in function_features.feature)
        expected_features = {
            'head_0_word+head_0_pos': 'özgürlüğünü_Noun',
            'head_0_word': 'özgürlüğünü',
            'head_0_pos': 'Noun',
            'child_0_word+child_0_pos': 'Kerem_Prop',
            'child_0_word': 'Kerem',
            'child_0_pos': 'Prop',
            'head_0_word+head_0_pos+child_0_word+child_0_pos': 'özgürlüğünü_Noun_Kerem_Prop',
            'head_0_pos+child_0_word+child_0_pos': 'Noun_Kerem_Prop',
            'head_0_word+child_0_word+child_0_pos': 'özgürlüğünü_Kerem_Prop',
            'head_0_word+head_0_pos+child_0_pos': 'özgürlüğünü_Noun_Prop',
            'head_0_word+head_0_pos+child_0_word': 'özgürlüğünü_Noun_Kerem',
            'head_0_word+child_0_word': 'özgürlüğünü_Kerem',
            'head_0_pos+child_0_pos': 'Noun_Prop',
            'head_0_pos+between_0_pos+child_0_pos': 'Noun_Punc_Prop',
            'head_0_pos+head_1_pos+child_-1_pos+child_0_pos': 'Noun_Noun_ROOT_Prop',
            'head_-1_pos+head_0_pos+child_-1_pos+child_0_pos': 'Punc_Noun_ROOT_Prop',
            'head_0_pos+head_1_pos+child_0_pos+child_1_pos': 'Noun_Noun_Prop_Punc',
            'head_-1_pos+head_0_pos+child_0_pos+child_1_pos': 'Punc_Noun_Prop_Punc',
            'head_0_word+head_0_up_word': 'özgürlüğünü_teslim',
            'head_0_pos+head_0_up_pos': 'Noun_Noun',
            'head_0_word+head_0_down_word': 'özgürlüğünü_None',
            'head_0_pos+head_0_down_pos': 'Noun_None',
            'child_0_word+child_0_up_word': 'Kerem_rahatlamıştı',
            'child_0_pos+child_0_up_pos': 'Prop_Verb',
            'child_0_word+child_0_down_word': 'Kerem_None',
            'child_0_pos+child_0_down_pos': 'Prop_None'
        }
        self.assertItemsEqual(features_dict, expected_features)
    
    def test_GetFeaturesWithExtendedSentence(self):
        
        head = self.test_sentence_en.token[2] # saw
        child = self.test_sentence_en.token[1] # john
        sentence = common.ExtendSentence(self.test_sentence_en)
        function_features = self.extractor.GetFeatures(sentence, head=head, child=child, use_tree_features=True)
        features_dict = dict((feature.name, feature.value) for feature in function_features.feature)
        expected_features = {
            'child_0_pos+child_0_down_pos':'Noun_None',
            'child_0_word+child_0_up_word':'John_saw',
            'head_0_word+head_0_pos+child_0_word':'saw_Verb_John',
            'child_0_word+child_0_pos':'John_Noun',
            'head_0_word':'saw',
            'child_0_word':'John',
            'head_0_pos+child_0_word+child_0_pos':'Verb_John_Noun',
            'head_0_pos+child_0_pos':'Verb_Noun',
            'head_0_word+head_0_down_word':'saw_Mary',
            'child_0_pos':'Noun',
            'head_0_pos+head_1_pos+child_0_pos+child_1_pos':'Verb_Noun_Noun_Verb',
            'head_0_word+head_0_pos':'saw_Verb',
            'head_0_pos+head_0_up_pos':'Verb_ROOT',
            'head_0_pos':'Verb',
            'head_0_pos+head_0_down_pos':'Verb_Noun',
            'head_0_pos+between_0_pos+child_0_pos':'Verb_None_Noun',
            'child_0_pos+child_0_up_pos':'Noun_Verb',
            'head_0_word+head_0_pos+child_0_word+child_0_pos':'saw_Verb_John_Noun',
            'head_-1_pos+head_0_pos+child_0_pos+child_1_pos':'Noun_Verb_Noun_Verb',
            'head_-1_pos+head_0_pos+child_-1_pos+child_0_pos':'Noun_Verb_ROOT_Noun',
            'head_0_word+head_0_up_word':'saw_ROOT',
            'head_0_word+child_0_word':'saw_John',
            'head_0_pos+head_1_pos+child_-1_pos+child_0_pos':'Verb_Noun_ROOT_Noun',
            'head_0_word+child_0_word+child_0_pos':'saw_John_Noun',
            'child_0_word+child_0_down_word':'John_None',
            'head_0_word+head_0_pos+child_0_pos':'saw_Verb_Noun',            
        }
        self.assertItemsEqual(features_dict, expected_features)

    
if __name__ == "__main__":
  unittest.main()