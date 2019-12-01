# -*- coding: utf-8 -*-

"""Dependency Labeler."""

import numpy as np
from learner.label_perceptron import LabelPerceptron
from data.treebank import sentence_pb2
from google.protobuf import text_format
from learner import featureset_pb2
from learner.feature_extractor import FeatureExtractor
from util import common
from util import reader
from util import writer

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)


# A dependency labeler works on the output of the dependency parser to label
# the dependency arcs assingned by the parser to the heads and dependents.
class DependencyLabeler:
    def __init__(self, feature_opts={}):
        self.feature_opts = feature_opts
        self.feature_extractor = FeatureExtractor("labelfeatures")
        self.label_perceptron = LabelPerceptron(self.feature_opts)
        self.label_accuracy_train = None
        self.label_accuracy_test = None

    def MakeFeatures(self, training_data):
        """Makes features from the training data.

        Note that for dependency labeler, the relevant features are features
        from the gold head-dependent pairs.
        Args:
            training_data: list, list of sentence_pb2.Sentence() objects.
        """
        self.label_perceptron.MakeAllFeatures(training_data)

    def PredictLabels(self, sentence):
        """Label the dependency arcs in a sentence.
        Args:
            sentence: sentence_pb2.Sentence(), without dependency labels.
        Returns:
            labeled: sentence_pb2.Sentence(), with dependency labels.
            predicted_labels = list, list of predicted labels.
        """
        assert sentence.HasField("length"), "Sentence must have length!!"
        predicted_labels = []
        for token in sentence.token:
            #print("token is {}".format(token.word))
            #print("selected head {}".format(token.selected_head.address))
            if token.selected_head.address == -1 or token.word == "ROOT":
              predicted_labels.append(u"")
              continue
            label, _, _ = self.label_perceptron.PredictLabel(sentence, token)
            predicted_labels.append(label)
        return predicted_labels
    
    def InsertLabels(self, sentence, labels):
      """Insert the predicted labels into the sentence.
      Args:
        sentence: sentence_pb2.Sentence().
        labels: list, list of labels predicted by the system.
      Returns:
        sentence: the sentence where the labels are inserted. 
      """
      assert len(sentence.token) == len(labels), "Mismatch between the number of tokens and labels!"
      for i, token in enumerate(sentence.token):
        token.ClearField("label")
        if token.selected_head.address == -1 or token.word == "ROOT":
          continue
        token.label = labels[i]
      return sentence
      
    def Train(self, niters, training_data, test_data=None, approx=10):
        """Train the label perceptron."""
        for i in range(niters):
            print("\n**************-------------------*************")
            logging.info("Starting LP Training Epoch {}".format(i+1))
            #Train label perceptron for one epoch.
            correct = self.label_perceptron.Train(training_data)
            #Evaluate the label perceptron
            train_acc = self.Evaluate(training_data)
            logging.info("LP train acc after iter {}: {}".format(i+1, train_acc))
            #raw_input("Press a key to continue: ")
            if test_data:
                test_acc = self.Evaluate(test_data, eval_type="test")
                # Comment out if you're not interested in seeing test acc after
                # each epoch.
                logging.info("LP Test acc after iter {}: {}".format(i+1, test_acc))
                #raw_input("Press a key to continue: ")
            #if train_acc == 100:
            #    break
            np.random.shuffle(training_data)

    def Evaluate(self, eval_data, eval_type="train"):
        """Evaluates the performance of label perceptron on data.
        Args:
            eval_data = list, list of sentence_pb2.Sentence() objects.
            eval_type = whether we evaluate on training or test data.
        Returns:
            accuracy of the label perceptron on the dataset.
        """
        assert eval_type in ["train", "test"], "Invalid eval type!!"
        if eval_type == "train":
          logging.info("Evaluating on training data")
        else:
          logging.info("Evaluating on test data")
        acc = 0.0
        #print(len(eval_data))
        for sentence in eval_data:
            #print(sentence.text)
            gold_labels = [token.label for token in sentence.token]
            assert sentence.HasField("length"), "Sentence must have a length!"
            #common.PPrintTextProto(sentence)
            # label the sentence with the model
            predicted_labels = self.PredictLabels(sentence)
            #predicted_labels = [token.label for token in sentence.token]
            #logging.info("predicted {}, gold {}".format(predicted_labels, gold_labels))
            assert len(predicted_labels) == len(gold_labels), """Number of
                predicted and gold labels don't match!!!"""
            acc += self._Accuracy(predicted_labels, gold_labels)
        if eval_type == "train":
          self.label_accuracy_train = acc / len(eval_data)
          return self.label_accuracy_train
        else:
          self.label_accuracy_test = acc / len(eval_data)
          return self.label_accuracy_test

    def _Accuracy(self, prediction, gold):
        return 100 * sum(pl == gl for pl, gl in zip(prediction, gold)) / len(prediction)

def main():
    labeler = DependencyLabeler()

if __name__ == "__main__":
    main()
