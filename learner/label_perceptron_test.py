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
        percept.label_weights[class_]["bias"]["bias"] = 2
        percept.label_weights[class_]["head_0_word"]["ROOT"] = 5
        features = percept._ConvertWeightsToProto(class_)
        self.assertEqual(7.0, percept.Score(class_, features))

        class_ = "nsubj"
        num_features = len(features.feature)
        weight = 1
        for key in percept.label_weights[class_].keys():
            for value in percept.label_weights[class_][key].keys():
                percept.label_weights[class_][key][value] += weight
                #print(key, value, percept.weights[key][value])
                weight += 1
        score = percept.Score(class_, features)
        # the gaussian formula for summing consecutive numbers
        gauss = num_features * (num_features + 1) / 2
        self.assertEqual(score, gauss)

        # make sure that the class whose features we didn't touch is still 0.
        self.assertEqual(0.0, percept.Score("parataxis", features))
        print("Passed!")


if __name__ == "__main__":
  unittest.main()
