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

class PerceptronTest(unittest.TestCase):
    """Tests for perceptron."""
    
    def setUp(self):
        self.en_test = _read_perceptron_test_data("john_saw_mary")
    
    
    def test_MakeFeaturesFromGold(self):
        pass
    '''
    def test_MakeAllFeatures(self):
        percept = perceptron.ArcPerceptron()
        percept.MakeAllFeatures([self.en_test])
        sorted_features = common.SortFeatures(percept._ConvertWeightsToProto())
        expected_features = _read_features_test_data("john_saw_mary_features")
        self.assertEqual(sorted_features, expected_features)
    '''    
    '''
    def test_Score(self):
        percept = perceptron.ArcPerceptron()
        percept.LoadFeatures("kerem_ozgurlugunu_load_test")
        self.assertEqual(15, percept._Score(percept._ConvertWeightsToProto()))
    '''
    '''
    def test_PredictHead(self):
        percept = perceptron.ArcPerceptron()
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
        self.assertEqual(scores, [1630, 1660, None, 1665])
        self.assertEqual(prediction, 3)
    '''
    def testTrain(self):
        percept = perceptron.ArcPerceptron()
        percept.MakeAllFeatures([self.en_test])
        init_w = 0.1
        for key in percept.weights.keys():
            for value in percept.weights[key].keys():
                percept.weights[key][value] += init_w
                #print(key, value, percept.weights[key][value])
                init_w += 0.1
        '''
        f_weights_before_train[saw_root] = GetFeatureWeights(extractor.GetFeatures(..))
        f_weights_after_train[saw_root] = GetFeatureWeights(extractor.GetFeatures(..))
        assertTrue(f_weights_after_train[saw_root] == map(lambda x:x+1, f_weights_before_train[saw_root]))
        '''
        percept.Train(1, [self.en_test])
    
    def test_LoadFeatures(self):
        percept = perceptron.ArcPerceptron()
        percept.LoadFeatures("kerem_ozgurlugunu_load_test", as_text=True)
        percept._ConvertWeightsToProto()
        
        expected_featureset = _read_features_test_data("kerem_ozgurlugunu")
        percept_fkeys = [f.name for f in percept.featureset.feature]
        expected_fkeys = [f.name for f in expected_featureset.feature]
        self.assertListEqual(sorted(percept_fkeys), sorted(expected_fkeys))
        
        percept_fvalues = [f.value for f in percept.featureset.feature]
        expected_fvalues = [f.value for f in expected_featureset.feature]
        self.assertListEqual(sorted(percept_fvalues), sorted(expected_fvalues))
        
        percept_fweights = [f.weight for f in percept.featureset.feature]
        expected_fweights = [f.weight for f in expected_featureset.feature]
        self.assertListEqual(sorted(percept_fweights), sorted(expected_fweights))
        
if __name__ == "__main__":
  unittest.main()