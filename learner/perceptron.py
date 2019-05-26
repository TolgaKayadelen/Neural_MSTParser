# -*- coding: utf-8 -*-

"""Averaged Perceptron Algorithm. 
The perceptron is a an online margin-based linear classifier."""

import argparse
import os
import pickle

from collections import defaultdict
from collections import OrderedDict
from copy import deepcopy
from data.treebank import sentence_pb2
from google.protobuf import text_format
from learner import featureset_pb2
from learner.feature_extractor import FeatureExtractor
from mst.max_span_tree import GetTokenByAddressAlt 
from util import common
from util import reader
from util import writer

_MODEL_DIR = "model/pretrained"

class AveragedPerceptron(object):
    """Base methods for the Averaged Perceptron."""
    def __init__(self):
        self.weights = defaultdict(OrderedDict)
        self._timestamps = defaultdict(OrderedDict)
        self._accumulator = defaultdict(OrderedDict)
        self._feature_count = 0
    
    def InitializeWeights(self, featureset, load=False):
        """Initialize the weights with zero or a pretrained value. 
        
        Args:
            featureset: featureset_pb2.FeatureSet(), set of features. 
            load: boolean, if True, loads a featureset with pretrained weights.
        """
        if load:
            for f in featureset.feature:
                self.weights[f.name].update({f.value:f.weight})
        else:
            for f in featureset.feature:
                self.weights[f.name].update({f.value:0.0})
        
        # compute the number of total features
        totals = [len(self.weights[key]) for key in self.weights.keys()]
        self._feature_count = sum(totals)

    
    def AverageWeights(self, weights):
        """Average the weights over all iterations.""" 
        assert self.iterations > 0, "Cannot average weights"
        for name, value in weights.iteritems():
            if isinstance(value, dict):
                self._AverageWeights(weights)
            else:
                weights[name] = weights[name] / self.iters
        del self._totals
        del self._timestamps
        return weights
    
    
    def _ConvertWeightsToProto(self):
        """Convert the weights dictionary to featureset proto.
        
        This method should be accessed either via SortFeatures() or TopFeatures().
        """
        assert not hasattr(self, "featureset"), "Weights already converted to proto in self.featureset"
        self.featureset = featureset_pb2.FeatureSet()
        for name, v in self.weights.iteritems():
            for value, weight in v.iteritems():
                feature = self.featureset.feature.add(
                    name=name,
                    value=value,
                    weight=weight,
                    )
        #print(text_format.MessageToString(featureset, as_utf8=True))
        error = "Number of converted features and total_features don't match."
        assert len(self.featureset.feature) == self._feature_count, error
        return self.featureset
        
        
    def LoadFeatures(self, filename, as_text=False):
        """Load pretrained features and their weights.
        
        Populates self.weights with the read object. The read object
        is a featureset_pb2.FeatureSet() proto.
        Args:
            filename: string, the filename to load the features from.
            as_text: boolean, if True, tries to read a .pbtxt file
        """
        if as_text:
            path = os.path.join(_MODEL_DIR, "{}.pbtxt".format(filename))
            with open(path, "r") as f:
                featureset = text_format.Parse(f.read(), featureset_pb2.FeatureSet())
        else:
            path = os.path.join(_MODEL_DIR, "{}.pkl".format(filename))
            with open(path, "rb") as inp:
                featureset = pickle.load(inp)
        print("features read from {}".format(path))
        self.InitializeWeights(featureset, load=True)

           
    def SaveFeatures(self, filename, as_text=False):
        """Save trained features and their weights to a .pkl file.
        
        The saved object is a featureset_pb2.FeatureSet proto.
        Args:
            weights = defaultdict, the feature weights to save.
            filename: string, path to save the trained features into.
        """            
        if not hasattr(self, "featureset"):
            self.featureset = self._ConvertWeightsToProto()
        
        assert filename, "No output filename!"
        
        if as_text:
            output_file = os.path.join(_MODEL_DIR, "{}.pbtxt".format(filename))
            writer.write_proto_as_text(self.featureset, output_file)
        else:
            output_file = os.path.join(_MODEL_DIR, "{}.pkl".format(filename))
            with open(output_file, 'wb') as output:
                pickle.dump(self.featureset, output)
        print("saved features to {}".format(output_file))       

    
    def Sort(self):
        """Sort features by weight."""
        #TODO: write a test
        error = "No featureset to sort, convert weights to featureset proto first."
        assert hasattr(self, "featureset"), error
        self.featureset = common.SortFeatures(self.featureset)
        return self.featureset

        
    def TopN(self, n):
        """Return the n features with the largest weight."""
        error = "No featureset to sort, convert weights to featureset proto first."
        assert hasattr(self, "featureset"), error
        return common.TopFeatures(self.featureset)
                    

class ArcPerceptron(AveragedPerceptron):
    """A perceptron for scoring dependency arcs."""
    
    def __init__(self, feature_options={}):
        super(ArcPerceptron, self).__init__()
        self.feature_options = feature_options
        self.iters = 0
        self._extractor = FeatureExtractor()
        
    def MakeFeaturesFromGold(self, training_data):
        """Create a feature set from the gold head-dependent pairs in the data.
        Args:
            training_data: list of sentence_pb2.Sentence() objects.
            TODO: in the future this will be a treebank proto. 
        """
        # assert isinstance(training_data, treebank_pb2.Treebank())
        for sentence in training_data:
            assert isinstance(sentence, sentence_pb2.Sentence), "Unaccapted data type."
            sentence = common.ExtendSentence(sentence)
            for token in sentence.token:
                # skip where token is the dummy start token, dummy end token, or the root token. 
                if not token.selected_head or token.selected_head.address == -1:
                    continue
                # just another safety check to make sure that we catch everything we need to skip.
                if token.index <= 0:
                    continue
                #print("child: {}".format(token.word))
                head = GetTokenByAddressAlt(sentence.token, token.selected_head.address)
                #print("head {}, child {}".format(head.word, token.word))
                self.InitializeWeights(self._extractor.GetFeatures(
                    sentence,
                    head=head,
                    child=token,
                    use_tree_features=True)
                )
    
    def MakeAllFeatures(self, training_data):
        """Create a features set from --all-- head-dependent pairs in the data.
        Args:
            training_data: list of sentence_pb2.Sentence() objects.
        """
        for sentence in training_data:
            assert isinstance(sentence, sentence_pb2.Sentence), "Unaccepted data type."
            sentence = common.ExtendSentence(sentence)
            for token in sentence.token:
                # skip where token is the dummy start token, dummy end token, or the root token. 
                if not token.selected_head or token.selected_head.address == -1:
                    continue
                # just another safety check to make sure that we catch everything we need to skip.
                if token.index <= 0:
                    continue
                # ch = candidate_head
                for ch in token.candidate_head:
                    head = GetTokenByAddressAlt(sentence.token, ch.address)
                    #print("head {}, child {}".format(head.word, token.word))
                    self.InitializeWeights(self._extractor.GetFeatures(
                        sentence,
                        head=head,
                        child=token,
                        use_tree_features=True)
                    )
   
    def _Score(self, features):
        """Score a feature vector.
        
        features = featureset_pb2.FeatureSet()
        """
        score = 0.0
        for feature in features.feature:
            if feature.name not in self.weights:
                #print("not in weights {}, {}".format(feature.name, feature.value))
                continue
            #print("feature is {}".format(self.weights[feature.name][feature.value]))
            score += self.weights[feature.name][feature.value]
        return score

    
    def _Predict(self, sentence, token, weights=None):
        """Greedy head prediction used for training.
        
        Dot product the features and current weights to return the best label.
        
        Args:
            sentence = sentence_pb2.Sentence()
            token = sentence_pb2.Token()
            weights = the weights dictionary.
        """
        scores = []
        features = []
        for head in sentence.token:
            featureset = self._extractor.GetFeatures(sentence, head, token, use_tree_features=True)
            score = self._Score(featureset)
            features.append(featureset)
            scores.append(score)
        prediction = np.argmax(scores)
        return prediction, features

    
    def UpdateWeights(self, prediction_features, true_features):
        """Update the feature weights based on prediction.
        
        If the active features give you the correct prediction, increase
        self.weights[feature.name][feature.value] by 1, else decrease them by 1.
        
        Args:
            true_features = featureset_pb2.FeatureSet()
            guess_features = featureset_pb2.FeatureSet()
        """
        def upd_feat(fname, fvalue, w):
            if fname not in self.weights:
                pass
            else:
                nr_iters_at_this_weight = self.iters - self._timestamps[f]
                self._accumulator[f] += nr_iters_at_this_weight + self.weights[f]
                self._weights[fname][fvalue] += w 
                self._timestamps[f] += self.iters
        
        self.iters += 1
        for feature in true_features.feature:
            upd_feat(feature.name, feature.value, 1.0)
        for feature in guess_features.feature:
            upd_feat(feature.name, feature.value -1.0)
        
        
def main():
    perceptron = ArcPerceptron()
    #extractor = FeatureExtractor(filename="./learner/features.txt")
    test_sentence = reader.ReadSentenceTextProto("./data/testdata/generic/john_saw_mary.pbtxt")
    #head = test_sentence.token[3]
    #child = test_sentence.token[1]
    #features = extractor.GetFeatures(test_sentence, head, child, use_tree_features=True)
    #perceptron.MakeFeaturesFromGold([test_sentence])
    perceptron.MakeAllFeatures([test_sentence])
    #top_features = perceptron.TopFeatures(5)
    #sorted_features = perceptron.SortFeatures()
    #print(perceptron.TopFeatures(10))
    #print(text_format.MessageToString(sorted_features, as_utf8=True))


if __name__ == "__main__":
    main()