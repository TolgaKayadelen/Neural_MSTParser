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
        self.test_treebank_tr = reader.ReadTreebankTextProto("./data/testdata/features/kerem.pbtxt")
        self.test_sentence_tr = self.test_treebank_tr.sentence[0]
        self.test_sentence_en = reader.ReadSentenceTextProto("./data/testdata/generic/john_saw_mary.pbtxt")
        self.extractor = FeatureExtractor()
        self.maxDiff = None

    def test_GetFeatures(self):
        print("Running test_GetFeatures..")
        head = self.test_sentence_tr.token[3]
        child = self.test_sentence_tr.token[1]
        # Initialize the extractor with a feature file
        function_features = self.extractor.GetFeatures(self.test_sentence_tr, head, child)
        #print(text_format.MessageToString(function_features, as_utf8=True))
        features_dict = dict((feature.name, feature.value) for feature in function_features.feature)
        #for k, v in features_dict.items():
        #    print(k.encode("utf-8"), v.encode("utf-8"))
        expected_features = {
            u'head_0_word+head_0_pos': u'özgürlüğünü_Noun',
            u'head_0_word': u'özgürlüğünü',
            u'head_0_pos': u'Noun',
            u'child_0_word+child_0_pos': u'Kerem_Prop',
            u'child_0_word': u'Kerem',
            u'child_0_pos': u'Prop',
            u'head_0_word+head_0_pos+child_0_word+child_0_pos': u'özgürlüğünü_Noun_Kerem_Prop',
            u'head_0_pos+child_0_word+child_0_pos': u'Noun_Kerem_Prop',
            u'head_0_word+child_0_word+child_0_pos': u'özgürlüğünü_Kerem_Prop',
            u'head_0_word+head_0_pos+child_0_pos': u'özgürlüğünü_Noun_Prop',
            u'head_0_word+head_0_pos+child_0_word': u'özgürlüğünü_Noun_Kerem',
            u'head_0_word+child_0_word': u'özgürlüğünü_Kerem',
            u'head_0_pos+child_0_pos': u'Noun_Prop',
            u'head_0_pos+between_0_pos+child_0_pos': u'Noun_Punc_Prop',
            u'head_0_pos+head_1_pos+child_-1_pos+child_0_pos': u'Noun_Noun_ROOT_Prop',
            u'head_-1_pos+head_0_pos+child_-1_pos+child_0_pos': u'Punc_Noun_ROOT_Prop',
            u'head_0_pos+head_1_pos+child_0_pos+child_1_pos': u'Noun_Noun_Prop_Punc',
            u'head_-1_pos+head_0_pos+child_0_pos+child_1_pos': u'Punc_Noun_Prop_Punc',
            u'child_0_lemma': u'Kerem',
            u'head_0_lemma': u'özgürlük',
            u'head_0_word+head_0_lemma': u'özgürlüğünü_özgürlük',
            u'child_0_word+child_0_lemma': u'Kerem_Kerem',
            u'head_0_pos+head_0_morph_case': u'Noun_acc',
            u'child_0_pos+child_0_morph_case': u'Prop_nom',
            u'child_0_morph_case': u'nom',
            u'head_0_morph_case': u'acc',
            u'head_0_pos+head_0_morph_case+child_0_pos+child_0_morph_case': u'Noun_acc_Prop_nom'
        }

        self.assertDictEqual(features_dict, expected_features)
        print("Passed!")

    def test_GetFeaturesWithExtendedSentence(self):
        print("Running testGetFeaturesWithExtendedSentence..")
        head = self.test_sentence_en.token[2] # saw
        child = self.test_sentence_en.token[1] # john
        sentence = common.ExtendSentence(self.test_sentence_en)
        function_features = self.extractor.GetFeatures(sentence, head=head, child=child)
        features_dict = dict((feature.name, feature.value) for feature in function_features.feature)
        expected_features = {
            u'head_0_word+head_0_pos+child_0_word': u'saw_Verb_John',
            u'child_0_word+child_0_pos': u'John_Noun',
            u'head_0_word': u'saw',
            u'child_0_word': u'John',
            u'head_0_pos+child_0_word+child_0_pos': u'Verb_John_Noun',
            u'head_0_pos+child_0_pos': u'Verb_Noun',
            u'child_0_pos': u'Noun',
            u'head_0_pos+head_1_pos+child_0_pos+child_1_pos': u'Verb_Noun_Noun_Verb',
            u'head_0_word+head_0_pos': u'saw_Verb',
            u'head_0_pos': u'Verb',
            u'head_0_pos+between_0_pos+child_0_pos': u'Verb_None_Noun',
            u'head_0_word+head_0_pos+child_0_word+child_0_pos': u'saw_Verb_John_Noun',
            u'head_-1_pos+head_0_pos+child_0_pos+child_1_pos': u'Noun_Verb_Noun_Verb',
            u'head_-1_pos+head_0_pos+child_-1_pos+child_0_pos': u'Noun_Verb_ROOT_Noun',
            u'head_0_word+child_0_word': u'saw_John',
            u'head_0_pos+head_1_pos+child_-1_pos+child_0_pos': u'Verb_Noun_ROOT_Noun',
            u'head_0_word+child_0_word+child_0_pos': u'saw_John_Noun',
            u'head_0_word+head_0_pos+child_0_pos': u'saw_Verb_Noun',
            u'child_0_lemma': u'John',
            u'head_0_lemma': u'see',
            u'head_0_word+head_0_lemma': u'saw_see',
            u'child_0_word+child_0_lemma': u'John_John',
            u'head_0_pos+head_0_morph_case': u'Verb_None',
            u'child_0_pos+child_0_morph_case': u'Noun_None',
            u'child_0_morph_case': u'None',
            u'head_0_morph_case': u'None',
            u'head_0_pos+head_0_morph_case+child_0_pos+child_0_morph_case': u'Verb_None_Noun_None'
        }

        self.assertDictEqual(features_dict, expected_features)
        print("Passed!")


if __name__ == "__main__":
  unittest.main()
