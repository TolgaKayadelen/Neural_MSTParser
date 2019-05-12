# -*- coding: utf-8 -*-

"""Averaged Perceptron Algorithm. 
The perceptron is a an online margin-based linear classifier."""

import argparse
import re

from collections import defaultdict
from collections import OrderedDict
from copy import deepcopy
from data.treebank import sentence_pb2
from learner import featureset_pb2
from learner.feature_extractor import FeatureExtractor 
from google.protobuf import text_format
from util import reader
from util import common

class AveragedPerceptron(object):
    """Base methods for the Averaged Perceptron."""
    def __init__(self):
        """Initialize the perceptron with a featureset"""
        pass
    
    def InitializeWeightsDict(self, featureset=None):
        """Initialize the weights with zero. 
        This function initializes three dictionaries, whose keys are feature_value pairs
        and whose values are weights. 
        
        featureset = featureset.FeatureSet() object. A proto of features.
        """
        self.weights = defaultdict(OrderedDict)
        # read features from the features proto to default dict.
        #w = 0.0
        for f in featureset.feature:
            self.weights[f.name].update({f.value:f.weights})
            #w += 1
        
        # Initialize an accumulator dictionary.
        self._totals = deepcopy(self.weights)
        
        # Initialize a timestamp dictionary to log when each feature is last updated. 
        self._timestamps = deepcopy(self.weights)
    
    def AverageWeights(self, weights):
        """Average the weights over all iterations.""" 
        assert self.iterations > 0, "Cannot average weights"
        for name, value in weights.iteritems():
            if isinstance(value, dict):
                self._AverageWeights(weights)
            else:
                weights[name] = weights[name] / self.iterations
        del self._totals
        del self._timestamps
        return weights
    
    def SortFeatures(self):
        """Sort features by weight"""
        if not hasattr(self, "featureset"):
            print("Converting weights dict to featureset proto first..")
            self.featureset = self._ConvertWeightsToProto()
        print("Sorting FeatureSet proto..")
        unsorted_list = []
        for feature in self.featureset.feature:
            unsorted_list.append(feature)
        sorted_list = sorted(unsorted_list, key=lambda f: f.weight, reverse=True)
        sorted_featureset = featureset_pb2.FeatureSet()
        for f in sorted_list:
            sorted_featureset.feature.add(name=f.name, value=f.value, weight=f.weight)
        del sorted_list
        del unsorted_list
        self.featureset.CopyFrom(sorted_featureset)
        return self.featureset
        
    def TopFeatures(self, n):
        """Return the n features with the largest weight."""
        if not hasattr(self, "featureset"):
            self.featureset = self._ConvertWeightsToProto()
            self.SortFeatures()
        return self.featureset.feature[:n]
        
    def _ConvertWeightsToProto(self):
        """Convert the weights dictionary to featureset proto.
        
        This method should be accessed either via SortFeatures() or TopFeatures().
        """
        assert not hasattr(self, "featureset"), "Weights already converted to proto in self.featureset"
        self.featureset = featureset_pb2.FeatureSet()
        for name, v in self.weights.iteritems():
            feature = self.featureset.feature.add(name=name)
            for value, weight in v.iteritems():
                feature.value = value
                feature.weight = weight
        #print(text_format.MessageToString(featureset, as_utf8=True))
        return self.featureset
                    
    # TODO: implement save and load methods. 
    
    
    
    

def main():
    perceptron = AveragedPerceptron()
    extractor = FeatureExtractor(filename="./learner/features.txt")
    test_sentence = reader.ReadSentenceProto("./data/treebank/sentence_4.protobuf")
    head = test_sentence.token[3]
    child = test_sentence.token[1]
    features = extractor.GetFeatures(test_sentence, head, child, use_tree_features=True)
    perceptron.InitializeWeightsDict(features)
    #top_features = perceptron.TopFeatures(5)
    sorted_features = perceptron.SortFeatures()
    print(perceptron.TopFeatures(10))
    #print(text_format.MessageToString(sorted_features, as_utf8=True))
    


if __name__ == "__main__":
    main()