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

class ArcPerceptronTest(unittest.TestCase):
    """Tests for arc perceptron."""

    def setUp(self):
        temp = _read_perceptron_test_data("john_saw_mary")
        self.en_test = common.ExtendSentence(temp)

    def test_MakeAllFeatures(self):
        print("Running test_MakeAllFeatures..")
        percept = perceptron.ArcPerceptron(feature_file="arcfeatures_base")
        percept.MakeAllFeatures([self.en_test])
        sorted_features = common.SortFeatures(percept._ConvertWeightsToProto())
        expected_features = _read_features_test_data("john_saw_mary_features")
        self.assertEqual(sorted_features, expected_features)
        print("Passed!")

    def testScore(self):
        print("Running testScore..")
        percept = perceptron.ArcPerceptron(feature_file="arcfeatures_base")
        path = os.path.join("data/testdata/features", "kerem_ozgurlugunu_score_test.pbtxt")
        with open(path, "r") as f:
            featureset = text_format.Parse(f.read(), featureset_pb2.FeatureSet())
        percept.InitializeWeights(featureset, load=True)
        self.assertEqual(15, percept.Score(percept._ConvertWeightsToProto()))
        print("Passed!")

    def test_PredictHead(self):
        print("Running test_PredictHead..")
        percept = perceptron.ArcPerceptron(feature_file="arcfeatures_base")
        percept.MakeAllFeatures([self.en_test])
        init_w = 1
        for key in percept.weights.keys():
            for value in percept.weights[key].keys():
                percept.weights[key][value] += init_w
                #print(key, value, percept.weights[key][value])
                init_w += 1
        test_token = self.en_test.token[3] # saw
        prediction, features, scores = percept._PredictHead(self.en_test, test_token)
        #print(scores)
        #print(prediction)
        self.assertEqual(scores, [1743.0, 1773.0, None, 1778.0])
        self.assertEqual(prediction, 3)
        print("Passed!")

    def testTrain(self):
        print("Running testTrain..")
        percept = perceptron.ArcPerceptron(feature_file="arcfeatures_base")
        weights_before_train = {}
        weights_after_train = {}
        percept.MakeAllFeatures([self.en_test])
        init_w = 0.1
        for key in percept.weights.keys():
            for value in percept.weights[key].keys():
                percept.weights[key][value] += init_w
                #print(key, value, percept.weights[key][value])
                init_w += 0.1
        #print(self.en_test)
        extractor = feature_extractor.FeatureExtractor(featuretype="arcfeatures", feature_file="arcfeatures_base")
        #head = Root, child=saw
        f_root_saw = extractor.GetFeatures(self.en_test, self.en_test.token[1], self.en_test.token[3])
        weights_before_train["root_saw"] = common.GetFeatureWeights(percept.weights, f_root_saw)
        # run one iteration of training.
        nr_correct, _ = percept.Train([self.en_test])
        weights_after_train["root_saw"] = common.GetFeatureWeights(percept.weights, f_root_saw)
        expected_weights_after_train = [
            11.1, 2.8, 7.4, 1.6, 2.3, 4.0, 6.4, 6.0, 5.9, 6.2, 14.2, 3.3, 1.1, 1.4, 15.4,
            8.7, 3.8, 13.6, 14.7, 1.7, 11.8, 4.7, 8.0, 12.7, 10.5, 5.4, 9.6]
        function_weights_after_train = [round(w, 1) for w in weights_after_train["root_saw"]]
        #print(function_weights_after_train)
        # Note that some weights looks like didn't change because in a later iteration
        # they caused wrong prediction and hence were reduced by 1.0 again.
        self.assertListEqual(function_weights_after_train, expected_weights_after_train)
        self.assertEqual(nr_correct, 1)
        print("Passed!")

    def testLearn(self):
        """Test that the perceptron actually learns."""
        print("Running testLearn..")
        percept = perceptron.ArcPerceptron(feature_file="arcfeatures_base")
        weights_before_train = {}
        weights_after_train = {}
        percept.MakeAllFeatures([self.en_test])
        init_w = 0.1
        for key in percept.weights.keys():
            for value in percept.weights[key].keys():
                percept.weights[key][value] += init_w
                #print(key, value, percept.weights[key][value])
                init_w += 0.1
        for i in range(3):
            nr_correct_heads, nr_childs = percept.Train([self.en_test])
            accuracy = nr_correct_heads * 100 / nr_childs
            print("accuracy after iter {} = %{}".format(i, accuracy))
            if accuracy == 100:
                break
        # we expect accuracy be 100 after the second iter.
        self.assertListEqual([i, accuracy], [1, 100])
        print("Passed!")

    def test_TimeStampsAndAveraging(self):
        """Test to make sure that timestamp dictionary is populated properly"""
        print("Running test_TimeStampsAndAveraging..")
        percept = perceptron.ArcPerceptron(feature_file="arcfeatures_base")
        percept.MakeAllFeatures([self.en_test])
        init_w = 0.1
        for key in percept.weights.keys():
            for value in percept.weights[key].keys():
                percept.weights[key][value] += init_w
                #print(key, value, percept.weights[key][value])
                init_w += 0.1
        # set up a feature for tracking when it's getting changed.
        feat_name = 'head_0_pos'
        feat_value = 'Verb'
        feat_timestamp = 0
        iters = 0
        for i in range(3):
            iters += 1
            feat_init_weight = percept.weights[feat_name][feat_value]
            #print("val at iter {}: {}".format(iters, feat_init_weight))
            nr_correct_heads, nr_childs = percept.Train([self.en_test])
            accuracy = nr_correct_heads * 100 / nr_childs
            if percept.weights[feat_name][feat_value] != feat_init_weight:
            #print("val changed to {} ".format(percept.weights[feat_name][feat_value]))
                feat_timestamp = iters
            if accuracy == 100:
                break

        self.assertEqual(44.0, round(percept._accumulator[feat_name][feat_value], 1))
        self.assertEqual(6, percept._timestamps[feat_name][feat_value])

        # test averaging
        acc_weights_for_feat = defaultdict(OrderedDict)
        acc_weights_for_feat[feat_name][feat_value] = percept._accumulator[feat_name][feat_value]
        averaged = percept.AverageWeights(acc_weights_for_feat)
        self.assertEqual(7.3, round(averaged[feat_name][feat_value], 1))
        print("Passed!")

    def testLoadModel(self):
        print("Running testLoadModel..")
        percept = perceptron.ArcPerceptron(feature_file="arcfeatures_base")
        percept.LoadModel("test_model", feature_file="arcfeatures_base" )
        self.assertEqual(percept.feature_count, 13360)
        print("Passed!")


if __name__ == "__main__":
  unittest.main()
