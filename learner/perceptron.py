# -*- coding: utf-8 -*-

"""Averaged Perceptron Algorithm.
The perceptron is a an online margin-based linear classifier."""

import argparse
import numpy as np
import os
import pickle
import random
import json

from collections import defaultdict
from collections import OrderedDict
from copy import deepcopy
from data.treebank import sentence_pb2
from google.protobuf import text_format
from google.protobuf import json_format
from learner import featureset_pb2
from learner.feature_extractor import FeatureExtractor
from util import common
from util import reader
from util import writer

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

_MODEL_DIR = "model/pretrained"

class AveragedPerceptron(object):
    """Base methods for the Averaged Perceptron."""
    def __init__(self):
        self.weights = defaultdict(OrderedDict)
        self._timestamps = defaultdict(OrderedDict)
        self._accumulator = defaultdict(OrderedDict)
        self.feature_count = 0

    def InitializeWeights(self, featureset, load=False):
        """Initialize the weights with zero or a pretrained value.

        Args:
            featureset: featureset_pb2.FeatureSet(), set of features.
            load: boolean, if True, loads a featureset with pretrained weights.
        """
        for f in featureset.feature:
            self._timestamps[f.name].update({f.value:0})
            self._accumulator[f.name].update({f.value:0})
            if load:
                self.weights[f.name].update({f.value:f.weight})
            else:
                self.weights[f.name].update({f.value:0.0})

        # compute the number of total features
        totals = [len(self.weights[key]) for key in self.weights.keys()]
        self.feature_count = sum(totals)


    def AverageWeights(self, weights):
        """Average the weights over all iterations."""
        assert self.iters > 0, "Cannot average weights"
        for name, value in weights.items():
            if isinstance(value, dict):
                self.AverageWeights(value)
            else:
                weights[name] = weights[name] / self.iters
        #del self._accumulator
        #del self._timestamps
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
        assert len(self.featureset.feature) == self.feature_count, error
        return self.featureset

    def LoadModel(self, name):
        """Load model features and weights from a json file.
        Args:
            name = the name of the model to load.
        """
        name = name + ".json" if not name.endswith(".json") else name
        input_file = os.path.join(_MODEL_DIR, "{}".format(name))
        with open(input_file, "r") as inp:
            model = json.load(inp)
        featureset = json_format.Parse(model["featureset"], featureset_pb2.FeatureSet())
        feature_options = model["feature_options"]
        accuracy = model["accuracy"]
        logging.info("Arc accuracy of the loaded model: {}".format(accuracy))
        logging.info("Feature options of the loaded model: {}".format(feature_options))
        self.InitializeWeights(featureset, load=True)

    def SaveModel(self, name, train_data_path=None, test_data_path=None,
    	 nr_epochs=None, accuracy=None):
        """Save model features and weights in json format.
        Args:
            name: string, the name of the model.
            data_path: string, the data path with which the model was trained.
            epocsh: the training epochs.
            accuracy: the arc accuracy.
        """
        if not hasattr(self, "featureset"):
            self.featureset = self._ConvertWeightsToProto()
        name = name + ".json" if not name.endswith(".json") else name
        model = {
            "train_data_path": train_data_path,
            "test_data_path": test_data_path,
            "epochs_trained": nr_epochs,
            "accuracy": accuracy,
            "feature_options": self.feature_options,
            "featureset": json_format.MessageToJson(self.featureset,
            	including_default_value_fields=True)
        }
        output_file = os.path.join(_MODEL_DIR, "{}".format(name))
        with open(output_file, "w") as output:
            json.dump(model, output, indent=4)
        logging.info("""Saved model with the following specs:
            train_data: {},
            test_data: {},
            epochs: {},
            accuracy: {},
            feature_options: {},
            feature_count: {}""".format(train_data_path, test_data_path, nr_epochs, accuracy,
                        self.feature_options, self.feature_count))

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
        self._extractor = FeatureExtractor("arcfeatures")

    def MakeAllFeatures(self, training_data):
        """Create a features set from --all-- head-dependent pairs in the data.
        Args:
            training_data: list of sentence_pb2.Sentence() objects.
        """
        for sentence in training_data:
            assert isinstance(sentence, sentence_pb2.Sentence), "Unexpected data type!!"
            assert len(sentence.token) == sentence.length + 2, "Unexpected sentence length!"
            for token in sentence.token:
                # skip where token is the dummy start token, dummy end token, or the root token.
                if not token.selected_head or token.selected_head.address == -1:
                    continue
                # just another safety check to make sure that we catch everything we need to skip.
                if token.index <= 0:
                    continue
                # ch = candidate_head
                for ch in token.candidate_head:
                    head = common.GetTokenByAddress(sentence.token, ch.address)
                    #print("head {}, child {}".format(head.word, token.word))
                    self.InitializeWeights(self._extractor.GetFeatures(
                        sentence,
                        head=head,
                        child=token)
                    )

    def Score(self, features):
        """Score a feature vector.

        features = featureset_pb2.FeatureSet()
        """
        score = 0.0
        for feature in features.feature:
            #if feature.name not in self.weights:
            if feature.value not in self.weights[feature.name]:
                #print("not in weights {}, {}".format(feature.name, feature.value))
                continue
            #print("feature is {} {}".format(feature.name, feature.value))
            score += self.weights[feature.name][feature.value]
            #print("weights for the features {}".format(self.weights[feature.name][feature.value]))
        return score


    def _PredictHead(self, sentence, token):
        """Greedy head prediction used for training.

        Dot product the features and current weights to return the best label.

        Args:
            sentence = sentence_pb2.Sentence()
            token = sentence_pb2.Token()
        Returns:
            prediction: int, index of the highest scoring features.
            features: list, list of featureset protos, used in prediction and scoring.
            scores: list, list of scores for features.
        """
        scores = []
        features = []
        for head in sentence.token:
            if head.word in ['START_TOK', 'END_TOK']:
                continue
            if head.index == token.index:
                # we don't make features for case where head = token
                # as no token can be its own head. So there's nothing to score.
                scores.append(None)
                features.append(None)
                continue
            #print("head: {}, child: {}".format(head.word, token.word))
            featureset = self._extractor.GetFeatures(sentence, head, token)
            score = self.Score(featureset)
            #print("score for this head-child {}".format(score))
            features.append(featureset)
            scores.append(score)
        # index of the highest scoring features
        prediction = np.argmax(scores)
        return prediction, features, scores


    def UpdateWeights(self, prediction_features, true_features, c=None):
        """Update the feature weights based on prediction.

        If the active features don't give you the correct prediction, increase
        self.weights[feature.name][feature.value] by 1 for the correct features
        and decrease self.weights[feature.name][feature.value] by 1 for the incorrect
        features. 

        Args:
            true_features = featureset_pb2.FeatureSet()
            guess_features = featureset_pb2.FeatureSet()
            c = whether the prediction was correct.

        NOTE: For debugging, turn on the commented out lines. This helps you track
        how a selected feature changes over time.
        """
        def upd_feat(fname, fvalue, w):
            if fname not in self.weights:
                logging.info("fname {}, passing this feature".format(fname))
                pass
            else:
                #print("updating the feature {}: {}".format(fname, fvalue))
                #print("which has the weight of {}".format(self.weights[feature.name][feature.value]))
                #if fname == 'head_0_pos' and fvalue == 'Verb':
                #    print("weight before ", self.weights[fname][fvalue])

                nr_iters_at_this_weight = self.iters - self._timestamps[fname][fvalue]

                #if fname == 'head_0_pos' and fvalue == 'Verb':
                #    print("nr_iters_here {}".format(nr_iters_at_this_weight))

                self._accumulator[fname][fvalue] += nr_iters_at_this_weight * self.weights[fname][fvalue]

                #if fname == 'head_0_pos' and fvalue == 'Verb':
                #    print("accumulated {}".format(self._accumulator[fname][fvalue]))

                self.weights[fname][fvalue] += w
                #print("updated value {}".format(self.weights[fname][fvalue]))
                self._timestamps[fname][fvalue] = self.iters

                #if fname == 'head_0_pos' and fvalue == 'Verb':
                #    print("weight after ", self.weights[fname][fvalue])
                #    print("timestamp for feat {}".format(self._timestamps[fname][fvalue]))
        # Since while predicting the head for each token for each sentence in
        # training data we call the update weights method, each token is one
        # iteration for the arc perceptron.
        self.iters += 1
        #print("-----")
        #print("iteration {}".format(self.iters))
        #if c:
        #    print("correct prediction")
        #else:
        #    print("incorrect prediction")
        for feature in true_features.feature:
            upd_feat(feature.name, feature.value, 1.0)
        for feature in prediction_features.feature:
            upd_feat(feature.name, feature.value, -1.0)

    def Train(self, training_data):
        """Trains arc perceptron for one epoch.

        Args:
            training_data: list, list of sentence_pb2.Sentence()

        Returns:
            correct: int, number of correct heads
            nr_child: number of arcs in the sentence.
        """
        correct = 0
        nr_child = 0
        for sentence in training_data:
            for token in sentence.token:
                # skip the dummy start and end tokens
                if token.word in ['START_TOK', 'END_TOK']:
                    continue
                # we don't try to predict a head for ROOT token.
                if token.word == "ROOT":
                    continue
                prediction, features, _ = self._PredictHead(sentence, token)
                #print("prediction {}".format(prediction))
                #print("selected head address {}".format(token.selected_head.address))
                #if not prediction == token.selected_head.address:
                c = prediction == token.selected_head.address
                self.UpdateWeights(features[prediction], features[token.selected_head.address], c)
                #TODO: do we need to return the correct and nr_child variables.
                correct += prediction == token.selected_head.address
                nr_child += 1
        #common.PPrintWeights(self._timestamps)
        return correct, nr_child

class LabelPerceptron(AveragedPerceptron):
    """A perceptron for labeling dependency arcs."""

    def __init__(self, feature_options={}):
        super(LabelPerceptron, self).__init__()
        self.feature_options = feature_options
        self.iters = 0
        self.labels = common.GetLabels().keys()
        self.label_weights = {}
        # private attributes
        self._extractor = FeatureExtractor("labelfeatures")
        self._label_timestamps = {}
        self._label_accumulator = {}
        self._label_count = len(self.labels)

    def _InitializeWeightsForEachClass(self):
        """Initializes separate weight vector for each class label.

        Args:
            featureset: featureset_pb2.FeatureSet(), set of features.
            load: boolean, if True, loads a featureset with pretrained weights.
        """
        for class_ in self.labels:
            self.label_weights[class_] = deepcopy(self.weights)
            self.label_weights[class_]["bias"].update({"bias":0.0})
            self._label_timestamps[class_] = deepcopy(self._timestamps)
            self._label_timestamps[class_]["bias"].update({"bias":0.0})
            self._label_accumulator[class_] = deepcopy(self._accumulator)
            self._label_accumulator[class_]["bias"].update({"bias":0.0})

        # Sanity check that the weight vectors are initialized properly for
        # labels
        random_class = random.choice(list(self.label_weights))
        #print(random_class)
        assert (self.feature_count == sum(
            [len(self.label_weights[random_class][key]) for key in self.label_weights[random_class].keys()]
                )-1
            ), "Mismatch between global and class specific label counts"

    def MakeAllFeatures(self, training_data):
        """Make features from gold head-dependency dependency pairs.
        Args:
            training_data: list of sentence_pb2.Sentence() objects.
        """
        for sentence in training_data:
            for token in sentence.token:
                if not token.selected_head or token.selected_head.address == -1:
                    continue
                head = common.GetTokenByAddress(
                    sentence.token, token.selected_head.address
                )
                #print("token {}".format(token))
                self.InitializeWeights(self._extractor.GetFeatures(
                    sentence,
                    head=head,
                    child=token
                    )
                )
        self._InitializeWeightsForEachClass()

    def Score(self, class_, features):
        """Score the feature vector for a class.
        Args:
            class_ = the class for which we are scoring the features.
            features = featureset_pb2.FeatureSet()
        """
        class_weights = self.label_weights[class_]
        # add the bias first.
        score = class_weights["bias"]["bias"]
        # add the other features.
        for feature in features.feature:
            if feature.value not in class_weights[feature.name]:
                continue
            score += class_weights[feature.name][feature.value]
        return score

    def PredictLabel(self, sentence, token):
        """Predict the best dependency label for a token.
        Args:
            token: sentence_pb2.Token()
        Returns:
            label: the label which has the highest score
        """
        best_score = -100000
        head = common.GetTokenByAddress(sentence.token, token.selected_head.address)
        #print("token is {}".format(token))
        #print("head is {}".format(head))
        features = self._extractor.GetFeatures(sentence, head, token)
        for class_ in self.labels:
            #print("scoring class: {}".format(class_))
            class_score = self.Score(class_, features)
            #print("class score: {}".format(class_score))
            #print("---")
            if class_score > best_score:
                best_score = class_score
                label = class_
        #print("best label: {}, best score: {}".format(label, best_score))
        return label, best_score


    def UpdateWeights(self, prediction, truth, features):
        """Update the feature weights based on prediction.
      
        If the active features don't give you the correct label, increase those
        features in self.label_weights[correct_label] by 1, and penalize the ones
        in self.label_weights[wrong_label] by -1.
        
        While doing weight update, you first increment iter, then increment accumulator,
        then increment the weight for the feature.
      
        Args:
          prediction: the predicted dependency label for the token.
          truth: the correct dependency label for the token.
          features: feature_set.pb2.FeatureSet(), the set of features to update.
        """
        def upt_features(class_, w, features):
          for f in features.feature:
              if f.name not in self.weights and f.name != "bias":
                  logging.info("fname {}, not in weights, passing".format(f.name))
              else:
                  nr_iters_at_weight = self.iters - self._label_timestamps[class_][f.name][f.value]
                  self._label_accumulator[class_][f.name][f.value] += nr_iters_at_weight * self.label_weights[class_][f.name][f.value]
                  self.label_weights[class_][f.name][f.value] += w
                  self._label_timestamps[class_][f.name][f.value] = self.iters
        upt_features(truth, 1.0, features)
        upt_features(prediction, -1.0, features)
        
    
    def UpdateAccumulator(self):
        """Update the accumulator for each weight in each class after each iteration.
        
        This function is for test purposes only and makes sure that the accumulator update
        from within the update weights function works as expected. This is a more expensive
        version of the accumulator update in UpdateWeights and should not be used other
        than for testing. 
      
        If you want to use this method, you should remove the label_accumulator update from
        the above function.
        """
        for class_ in self.labels:
            for key in self._label_accumulator[class_].keys():
                for value in self._label_accumulator[class_][key].keys():
                    self._label_accumulator[class_][key][value] += self.label_weights[class_][key][value]
    
    def IncrementIters(self):
        """Increments the number of iterations by 1.
      
        During training we call the IncrementIters method for each token in
        each sentence, therefore each token is one iteration for the Label
        Perceptron.
        """
        self.iters += 1

    def _ConvertWeightsToProto(self, class_, type_="weights"):
        """Convert a weights vector for a class to featureset proto.

        Overwrites the method in parent class.
        Args:
            class_: the class whose features we are converting.
            type_: label_weights, label_accumulator, or label_timestamps
        """
        assert type_ in ["weights", "accumulator", "timestamps"], "Invalid argument!!"
        if type_ == "weights":
            weights = self.label_weights
        elif type_ == "accumulator":
            weights = self._label_accumulator
        else:
            weights = self._label_timestamps
        featureset = featureset_pb2.FeatureSet()
        for name, v in weights[class_].iteritems():
            for value, weight in v.iteritems():
                feature = featureset.feature.add(
                    name=name,
                    value=value,
                    weight=weight,
                )
        #print(text_format.MessageToString(featureset, as_utf8=True))
        return featureset




def main():
    test_sentence = reader.ReadSentenceTextProto("./data/testdata/perceptron/john_saw_mary.pbtxt")
    arc_percp = ArcPerceptron()
    arc_percp.MakeAllFeatures([common.ExtendSentence(test_sentence)])
    for key, value in arc_percp.weights.items():
        print(key, value)
        print("----------------------------------")
    label_percp = LabelPerceptron()
    label_percp.MakeAllFeatures([test_sentence])
    for key, value in label_percp.label_weights.items():
        print(key, value)
        print("----------------------------------")

if __name__ == "__main__":
    main()
