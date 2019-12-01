# -*- coding: utf-8 -*-

"""A multiclass averaged perceptron used in label prediction."""

import argparse
import numpy as np
import os
import random


from collections import defaultdict
from collections import OrderedDict
from copy import deepcopy
from data.treebank import sentence_pb2
from google.protobuf import text_format
from google.protobuf import json_format
from learner import featureset_pb2
from learner.feature_extractor import FeatureExtractor
from learner.perceptron import AveragedPerceptron
from util import common
from util import reader
from util import writer

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)


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

        # Sanity check that weight vectors are initialized properly for labels.
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
        # add the bias weight first.
        score = class_weights["bias"]["bias"]
        # add other feature weights.
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
            label: the label which has the highest score.
            features: featureset_pb2.Featureset(), features used in scoring and prediction.
            best_score: the score for the winning class.
        """
        best_score = -100000
        head = common.GetTokenByAddress(sentence.token, token.selected_head.address)
        features = self._extractor.GetFeatures(sentence, head, token)
        for class_ in self.labels:
            class_score = self.Score(class_, features)
            if class_score > best_score:
                best_score = class_score
                label = class_
        return label, features, best_score


    def UpdateWeights(self, prediction, truth, features):
        """Update the feature weights based on prediction.
      
        If the active features don't give you the correct label, increase those
        features in self.label_weights[correct_label] by 1, and penalize the ones
        in self.label_weights[wrong_label] by -1. While doing weight update, you
        first increment iter, than increment accumulator, then increment the weight
        for the feature.
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
              #print("feature is \n{}".format(f))
              nr_iters_at_weight = self.iters - self._label_timestamps[class_][f.name][f.value]
                  
              # uncomment if you want to see testing logs
              #if class_ == "nsubj" and f.name == 'head_0_word+head_0_pos' and f.value == 'ROOT_ROOT':
              #    print("nr_iters_here {}".format(nr_iters_at_weight))
              #    print("weights at these iters {}".format(self.label_weights[class_][f.name][f.value]))
              #    print("accumulated so far {}".format(self._label_accumulator[class_][f.name][f.value]))

              self._label_accumulator[class_][f.name][f.value] += (
                nr_iters_at_weight * self.label_weights[class_][f.name][f.value])
              #if class_ == "nsubj" and f.name == 'head_0_word+head_0_pos' and f.value == 'ROOT_ROOT':
              #    print("new accum for weight {}".format(self._label_accumulator[class_][f.name][f.value]))

              temp = self.label_weights[class_][f.name][f.value]
              #if class_ == "nsubj" and f.name == 'head_0_word+head_0_pos' and f.value == 'ROOT_ROOT':
              #  print("feature updated from {} -> {}".format(temp, temp+w))
              #  print("-----")
  
              self.label_weights[class_][f.name][f.value] += w
              self._label_timestamps[class_][f.name][f.value] = self.iters

        upt_features(truth, 1.0, features)
        upt_features(prediction, -1.0, features)

    def IncrementIters(self):
        """Increments the number of iterations by 1.
      
        During training we call the IncrementIters() method for each token in
        each sentence, therefore each token is one iteration for the Label
        Perceptron.
        """
        self.iters += 1

    def Train(self, training_data):
        """Trains label perceptron for one epoch.
        Args: 
          training_data: list of sentence_pb2.Sentence.
        Returns: 
          correct: the number of correct labels in the training data.
        """
        correct = 0
        for sentence in training_data:
          for token in sentence.token:
            if token.word.lower() == "root" or token.selected_head.address == -1 or token.index == 0:
              continue
            self.IncrementIters()
            prediction, features, _ = self.PredictLabel(sentence, token)
            # Make sure to add the bias to the features as well. 
            features.feature.add(name="bias", value="bias")
            correct += prediction == token.label
            if prediction != token.label:
              #print("prediction is {}, true label is {}".format(prediction, token.label))
              self.UpdateWeights(prediction=prediction, truth=token.label, features=features)
        return correct
    
    def AverageClassWeights(self):
        for class_ in self.labels:
          for key in self.label_weights[class_].keys():
            for value in self.label_weights[class_][key].keys():
              # if class_ == "nsubj" and key == "head_0_pos" and value == "Verb":
              #  print("weight before {}".format(self.label_weights[class_][key][value]))
              self.label_weights[class_][key][value] = self._label_accumulator[class_][key][value] / self.iters
              # if class_ == "nsubj" and key == "head_0_pos" and value == "Verb":
              #  print("accumulated weight {}".format(self._label_accumulator[class_][key][value]))
              #  print("total iters = {}".format(self.iters))
              #  print("weight after = {}".format(self.label_weights[class_][key][value]))
              #  raw_input("Press a key to coninue..")
          
    
    def FinalizeAccumulator(self):
      """Finalizes the accumulator values for all features at the end of training.
      
      The client module that controls the training should call this function after 
      training is over. 
      """
      for class_ in self.labels:
        for key in self._label_accumulator[class_].keys():
          for value in self._label_accumulator[class_][key].keys():
            nr_iters_at_weight = self.iters - self._label_timestamps[class_][key][value]
            self._label_accumulator[class_][key][value] += (
              nr_iters_at_weight * self.label_weights[class_][key][value])
            # update the timestamps to show that the latest update to the 
            # accumulator has been done.
            # TODO: consider if this is optional or not.
            self._label_timestamps[class_][key][value] = self.iters
  
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
