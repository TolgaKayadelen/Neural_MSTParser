# -*- coding: utf-8 -*-

"""Unit tests for feature_extractor module"""

import unittest

from data.treebank import sentence_pb2
from google.protobuf import text_format
from learner.feature_extractor import FeatureExtractor
from util import reader
from util import common

class ArcFeatureExtractorTest(unittest.TestCase):
    """Tests for feature extractor"""

    def setUp(self):
        # Initialize a base document and a feature set for testing
        self.test_treebank_tr = reader.ReadTreebankTextProto("./data/testdata/features/kerem.pbtxt")
        self.test_sentence_tr = self.test_treebank_tr.sentence[0]
        self.test_sentence_en = reader.ReadSentenceTextProto("./data/testdata/generic/john_saw_mary.pbtxt")
        self.extractor = FeatureExtractor(featuretype="arcfeatures", feature_file=None, test=True)
        self.maxDiff = None

    def test_GetArcFeatures(self):
        print("Running test_GetArcFeatures..")
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

    def test_GetArcFeaturesWithExtendedSentence(self):
        print("Running testArcGetFeaturesWithExtendedSentence..")
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


class LabelFeatureExtractorTest(unittest.TestCase):
    """Tests for feature extractor"""

    def setUp(self):
        # Initialize a base document and a feature set for testing
        self.test_treebank_tr = reader.ReadTreebankTextProto("./data/testdata/features/kerem.pbtxt")
        self.test_sentence_tr = self.test_treebank_tr.sentence[0]
        self.test_sentence_en = reader.ReadSentenceTextProto("./data/testdata/generic/john_saw_mary.pbtxt")
        self.extractor = FeatureExtractor(featuretype="labelfeatures", feature_file=None, test=True)
        self.maxDiff = None

    def test_GetLabelFeatures(self):
        print("Running test_GetLabelFeatures..")
        head = self.test_sentence_tr.token[3]
        child = self.test_sentence_tr.token[1]
        # Initialize the extractor with a feature file
        function_features = self.extractor.GetFeatures(self.test_sentence_tr, head, child)
        #print(text_format.MessageToString(function_features, as_utf8=True))
        features_dict = dict((feature.name, feature.value) for feature in function_features.feature)
        expected_features = {
          u"child_0_pos+child_0_down_pos": u"Prop_None",
          u"head_0_morph_number": u"sing",
          u"child_0_lemma+head_0_lemma": u"Kerem_özgürlük",
          u"head_0_pos+head_0_morph_case": u"Noun_acc",
          u"child_0_pos+child_0_morph_case": u"Prop_nom",
          u"child_0_word+child_0_pos": u"Kerem_Prop",
          u"head_0_word": u"özgürlüğünü",
          u"child_0_word": u"Kerem",
          u"child_0_morph_number": u"sing",
          u"head_0_morph_number[psor]": u"sing",
          u"head_0_category+child_0_category": u"NOUN_PROPN",
          u"child_0_morph_case": u"nom",
          u"child_0_category+child_0_pos": u"PROPN_Prop",
          u"child_0_morph_number[psor]": u"None",
          u"child_0_pos": u"Prop",
          u"head_0_lemma": u"özgürlük",
          u"head_0_verbform": u"None",
          u"head_0_category+head_0_up_category": u"NOUN_NOUN",
          u"head_0_morph_person": u"3",
          u"head_0_category+head_0_pos": u"NOUN_Noun",
          u"head_0_pos+head_0_up_pos": u"Noun_Noun",
          u"child_0_morph_person": u"3",
          u"child_0_lemma": u"Kerem",
          u"head_0_pos": u"Noun",
          u"child_0_pos+head_0_pos+head_1_pos": u"Prop_Noun_Noun",
          u"child_0_voice": u"None",
          u"child_0_verbform": u"None",
          u"child_0_lemma+head_0_lemma+head_1_lemma": u"Kerem_özgürlük_teslim",
          u"head_0_word+head_0_pos": u"özgürlüğünü_Noun",
          u"head_0_morph_person[psor]": u"3",
          u"head_0_morph_case": u"acc",
          u"head_0_voice": u"None",
          u"head_0_category": u"NOUN",
          u"child_0_category": u"PROPN",
          u"child_0_category+child_0_down_category": u"PROPN_None",
          
        }
        self.assertDictEqual(features_dict, expected_features)
        
        head_1 = self.test_sentence_tr.token[9] # .
        child_1 = self.test_sentence_tr.token[8] # rahatlamisti
        function_features_1 = self.extractor.GetFeatures(self.test_sentence_tr, head_1, child_1)
        features_dict_1 = dict((feature.name, feature.value) for feature in function_features_1.feature)
        expected_features_1 = {
          u"child_0_pos+child_0_down_pos": u"Verb_Punc",
          u"head_0_morph_number": u"None",
          u"child_0_lemma+head_0_lemma": u"rahatla_.",
          u"head_0_pos+head_0_morph_case": u"Punc_None",
          u"child_0_pos+child_0_morph_case": u"Verb_None",
          u"child_0_word+child_0_pos": u"rahatlamıştı_Verb",
          u"head_0_word": u".",
          u"child_0_word": u"rahatlamıştı",
          u"child_0_morph_number": u"sing",
          u"head_0_morph_number[psor]": u"None",
          u"head_0_category+child_0_category": u"PUNCT_VERB",
          u"child_0_morph_case": u"None",
          u"child_0_category+child_0_pos": u"VERB_Verb",
          u"child_0_morph_number[psor]": u"None",
          u"child_0_pos": u"Verb",
          u"head_0_lemma": u".",
          u"head_0_verbform": u"None",
          u"head_0_category+head_0_up_category": u"PUNCT_VERB",
          u"head_0_morph_person": u"None",
          u"head_0_category+head_0_pos": u"PUNCT_Punc",
          u"head_0_pos+head_0_up_pos": u"Punc_Verb",
          u"child_0_morph_person": u"3",
          u"child_0_lemma": u"rahatla",
          u"head_0_pos": u"Punc",
          u"child_0_pos+head_0_pos+head_1_pos": u"Verb_Punc_None",
          u"child_0_voice": u"None",
          u"child_0_verbform": u"None",
          u"child_0_lemma+head_0_lemma+head_1_lemma": u"rahatla_._None",
          u"head_0_word+head_0_pos": u"._Punc",
          u"head_0_morph_person[psor]": u"None",
          u"head_0_morph_case": u"None",
          u"head_0_voice": u"None",
          u"head_0_category": u"PUNCT",
          u"child_0_category": u"VERB",
          u"child_0_category+child_0_down_category": u"VERB_PUNCT",
        }
        self.assertDictEqual(features_dict_1, expected_features_1)
        print("Passed!")
        
        #for k, v in features_dict_1.items():
        #  print "u"+'"'+k+'"'+":", "u"+'"'+v.encode("utf-8")+'"'+"," 



if __name__ == "__main__":
  unittest.main()
