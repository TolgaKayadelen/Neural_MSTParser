import unittest
import os

from collections import defaultdict
from collections import OrderedDict
from data.treebank import sentence_pb2
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

class PerceptronTest(unittest.TestCase):
    """Tests for perceptron."""
    
    def setUp(self):
        self.en_test = _read_perceptron_test_data("john_saw_mary")
    
    
    def test_MakeFeaturesFromGold(self):
        pass
        #TODO
    
    def test_MakeAllFeatures(self):
        percept = perceptron.ArcPerceptron()
        percept.MakeAllFeatures([self.en_test])
        sorted_features = common.SortFeatures(percept._ConvertWeightsToProto())
        expected_features = _read_features_test_data("john_saw_mary_features")
        self.assertEqual(sorted_features, expected_features)
        
    
    def test_Score(self):
        percept = perceptron.ArcPerceptron()
        percept.LoadFeatures("kerem_ozgurlugunu_load_test")
        self.assertEqual(15, percept._Score(percept._ConvertWeightsToProto()))
    
    def test_Predict(self):
        pass
        #percept = perceptron.ArcPerceptron()
        #percept.MakeAllFeatures([test_sentence])
    
    def test_LoadFeatures(self):
        percept = perceptron.ArcPerceptron()
        percept.LoadFeatures("kerem_ozgurlugunu_load_test", as_text=True)
        percept._ConvertWeightsToProto()
        
        expected_featureset = _read_features_test_data("kerem_ozgurlugunu")
        percept_keys = [f.name for f in percept.featureset.feature]
        expected_keys = [f.name for f in expected_featureset.feature]
        self.assertListEqual(sorted(percept_keys), sorted(expected_keys))
        
        percept_values = [f.value for f in percept.featureset.feature]
        expected_values = [f.value for f in expected_featureset.feature]
        self.assertListEqual(sorted(percept_values), sorted(expected_values))
        
        percept_weights = [f.weight for f in percept.featureset.feature]
        expected_weights = [f.weight for f in expected_featureset.feature]
        self.assertListEqual(sorted(percept_weights), sorted(expected_weights))
        
if __name__ == "__main__":
  unittest.main()