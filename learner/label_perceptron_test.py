# -*- coding: utf-8 -*-

import unittest
import os

from collections import defaultdict
from collections import OrderedDict
from data.treebank import sentence_pb2
from learner import feature_extractor
from learner import featureset_pb2
from learner import perceptron
from google.protobuf import text_format
from util import common

_TESTDATA_DIR = "data/testdata"
_FEATURES_DIR = os.path.join(_TESTDATA_DIR, "features")
_PERCEPTRON_DIR = os.path.join(_TESTDATA_DIR, "perceptron")

def _read_file(path):
    with open(path, "r") as f:
        read = f.read()
    return read

def _read_features_test_data(basename):
    path = os.path.join(_FEATURES_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), featureset_pb2.FeatureSet())

def _read_perceptron_test_data(basename):
    path = os.path.join(_PERCEPTRON_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())

class LabelPerceptronTest(unittest.TestCase):
    """Tests for label perceptron."""

    def setUp(self):
        self.en_test = _read_perceptron_test_data("john_saw_mary")

    def test_MakeAllFeatures(self):
        import random
        print("Running test_MakeAllFeatures..")
        percept = perceptron.LabelPerceptron()
        percept.MakeAllFeatures([self.en_test])
        random_class = random.choice(list(percept.label_weights))
        sorted_features = common.SortFeatures(percept._ConvertWeightsToProto(
            random_class), sort_key=lambda f: f.name)
        #print(sorted_features)
        expected_features = _read_features_test_data("john_saw_mary_label_features")
        expected_features.feature.add(
            name="bias",
            value="bias",
            weight=0.0
        )
        expected_features = common.SortFeatures(
            expected_features, sort_key=lambda f: f.name
        )
        self.assertEqual(sorted_features, expected_features)
        self.assertEqual(
            len(set([feature.name for feature in sorted_features.feature])),
            len(percept.label_weights[random_class].keys())
            )
        print("Passed!")

    def testScore(self):
        print("Running testScore..")
        percept = perceptron.LabelPerceptron()
        percept.MakeAllFeatures([self.en_test])
        class_ = "cc"
        percept.label_weights[class_]["head_0_word"]["ROOT"] = 5
        percept.label_weights[class_]["child_0_word"]["saw"] = 2
        features = percept._ConvertWeightsToProto(class_)
        self.assertEqual(7.0, percept.Score(class_, features))

        class_ = "nsubj"
        token = self.en_test.token[1] # john
        features = percept._extractor.GetFeatures(
            self.en_test,
            child=token,
            head=common.GetTokenByAddress(
                self.en_test.token,
                token.selected_head.address))
        num_features = len(features.feature)
        #print(features)
        weight = 1
        for key in percept.label_weights[class_].keys():
            for value in percept.label_weights[class_][key].keys():
                percept.label_weights[class_][key][value] += weight
                #print(key, value, percept.weights[key][value])
                weight += 1
        score = percept.Score(class_, features)

        expected = 0.0
        # add the bias first.
        expected += percept.label_weights[class_]["bias"]["bias"]
        # add the other feature weights
        for feature in features.feature:
            expected += percept.label_weights[class_][feature.name][feature.value]
        self.assertEqual(score, expected)

        # make sure that the class whose features we didn't touch is still 0.
        self.assertEqual(0.0, percept.Score("parataxis", features))
        print("Passed!")

    def testPredictLabel(self):
        print("Running testPredictLabel..")
        percept = perceptron.LabelPerceptron()
        percept.MakeAllFeatures([self.en_test])
        class_ = "cc"
        percept.label_weights[class_]["bias"]["bias"] = 2
        token = self.en_test.token[1]
        label, _ , score = percept.PredictLabel(self.en_test, token)
        self.assertEqual((label, score), ("cc", 2.0))
        print("Passed!!")

    def testUpdateWeights(self):
        print("Running testUpdateWeights")
        percept = perceptron.LabelPerceptron()
        percept.MakeAllFeatures([self.en_test])
        predicted = "cc"
        truth = "parataxis"
        token = self.en_test.token[1]
        head = common.GetTokenByAddress(self.en_test.token, token.selected_head.address)
        features = percept._extractor.GetFeatures(self.en_test, head=head, child=token)
        # make sure to add the bias to the features as well. 
        features.feature.add(name="bias", value="bias")   
        #print(features)        
        for i in range(5):
            percept.IncrementIters()
            #print("iteration {}".format(percept.iters))
            #percept.UpdateAccumulator()
            percept.UpdateWeights(predicted, truth, features)        
        # Uncomment if needed.
        #prediction_class_features = percept._ConvertWeightsToProto(predicted)
        #prediction_class_accumulator = percept._ConvertWeightsToProto(predicted, "accumulator")
        #prediction_class_tmstamp = percept._ConvertWeightsToProto(predicted, "timestamps")
        
        # check true class weights
        self.assertEqual(percept.label_weights[truth]["bias"]["bias"], 5.0)
        self.assertEqual(percept.label_weights[truth]["head_0_pos"]["Verb"], 5.0)
        self.assertEqual(percept._label_timestamps[truth]["head_0_pos"]["Verb"], 5)
        self.assertEqual(percept._label_accumulator[truth]["head_0_pos"]["Verb"], 10.0)
        self.assertEqual(percept.label_weights[truth]["child_0_word"]["Mary"], 0.0)
        
        # check predicted class weights
        self.assertEqual(percept.label_weights[predicted]["bias"]["bias"], -5.0)
        self.assertEqual(percept.label_weights[predicted]["head_0_pos"]["Verb"], -5.0)
        self.assertEqual(percept._label_timestamps[predicted]["head_0_pos"]["Verb"], 5)
        self.assertEqual(percept._label_accumulator[predicted]["head_0_pos"]["Verb"], -10.0)
        self.assertEqual(percept.label_weights[predicted]["child_0_word"]["Mary"], 0.0)
        print("Passed!!")

    def testTrain(self):
        print("Running testTrain")
        percept = perceptron.LabelPerceptron()
        percept.MakeAllFeatures([self.en_test])
        #common.PPrintWeights(percept.label_weights["cc"])
        
        # add the labels to the sentence.
        correct = 0
        tokens = self.en_test.token
        labels = ["nsubj", "root", "obj"]
        for i, token in enumerate(tokens[1:]):
          token.label = labels[i]
        for i in range(3):
          correct = percept.Train([self.en_test])
        tracked_feat = percept.label_weights["nsubj"]
        tracked_feat_tmstamp = percept._label_timestamps["nsubj"]
        self.assertEqual(correct, 3) # all labels are correctly predicted.
        self.assertEqual(percept.iters, 9)
        self.assertEqual(tracked_feat["child_0_lemma"]["John"], 2.0)
        self.assertEqual(tracked_feat["head_0_lemma"]["ROOT"], -1)
        self.assertEqual(tracked_feat["bias"]["bias"], 0.0)
        self.assertEqual(tracked_feat_tmstamp["child_0_lemma"]["John"], 4)
        self.assertEqual(tracked_feat_tmstamp["head_0_lemma"]["ROOT"], 2)
        self.assertEqual(tracked_feat_tmstamp["bias"]["bias"], 4)
        print("Passed!!")

    def testTimeStampsAndAveraging(self):
        print("Running testTimeStampsAndAveraging..")
        
        # initialize some weights
        percept = perceptron.LabelPerceptron()
        percept.MakeAllFeatures([self.en_test])
        class_ = "nsubj"
        feat_name = "head_0_word+head_0_pos"
        feat_value = "ROOT_ROOT"
        feat_timestamp = 0
        tokens = self.en_test.token
        labels = ["nsubj", "root", "obj"]
        for i, token in enumerate(tokens[1:]):
          token.label = labels[i]
        
        init_w = 0.1
        for key in percept.label_weights[class_].keys():
          for value in percept.label_weights[class_][key].keys():
            percept.label_weights[class_][key][value] += init_w
            init_w += 0.1
        
        # set up a feature for tracking when it is changed.
       
        for i in range(3):
          #print("accumulator at {}".format(i))
          #print(percept._label_accumulator[class_][feat_name][feat_value])
          #print("----")
          feat_init_weight = percept.label_weights[class_][feat_name][feat_value]
          cr = percept.Train([self.en_test])
          if percept.label_weights[class_][feat_name][feat_value] != feat_init_weight:
            print("Weight changed from {} -> {} at iter {}".format(
              feat_init_weight, percept.label_weights[class_][feat_name][feat_value], percept.iters
            ))
            feat_timestamp = percept.iters
            #print(feat_timestamp)
        #print(percept.iters)
        #print(percept.label_weights[class_][feat_name][feat_value])
        print(percept._label_accumulator[class_][feat_name][feat_value])
        print("Passed!!")



if __name__ == "__main__":
  unittest.main()
