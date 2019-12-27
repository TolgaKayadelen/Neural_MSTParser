# -*- coding: utf-8 -*-

"""Dependency Parser."""

import numpy as np
from learner.perceptron import ArcPerceptron
from data.treebank import sentence_pb2
from decoder import Decoder
from google.protobuf import text_format
from learner import featureset_pb2
from learner.feature_extractor import FeatureExtractor
from util import common
from util import reader
from util import writer

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

class DependencyParser:
    def __init__(self, feature_file=None, decoding="mst"):
        self.arc_perceptron = ArcPerceptron(feature_file)
        self.decoder = Decoder(decoding)
        self.train_accuracy = None
        self.test_accuracy= None

    def MakeFeatures(self, training_data):
        """Makes all features from the training data.

        Args:
            training_data: list, list of sentence_pb2.Sentence() objects.
        """
        self.arc_perceptron.MakeAllFeatures(training_data)

    def Parse(self, sentence):
        """Parse a sentence.

        Args:
            sentence: sentence_pb2.Sentence(), without head-dependent relations.

        Returns:
            parsed: sentence_pb2.Sentence(), parsed sentence.
            predicted_heads: list, list of predicted heads.
        """
        assert sentence.token[0].index == -1 and sentence.token[-1].index == -2
        assert sentence.HasField("length"), "Sentence must have a length."
        score_matrix = np.zeros((sentence.length, sentence.length))
        for token in sentence.token:
            #print("token is {}".format(token.word.encode("utf-8")))
            for ch in token.candidate_head:
                head = common.GetTokenByAddress(sentence.token, ch.address)
                #print("candidate head is {}".format(head.word.encode("utf-8")))
                features = self.arc_perceptron.feature_extractor.GetFeatures(
                    sentence = sentence,
                    head = head,
                    child = token
                )
                score = self.arc_perceptron.Score(features)
                score_matrix[token.index][head.index] = score
        #probs = self._Softmax(score_matrix)
        parsed, predicted_heads = self.decoder(sentence, score_matrix)
        return parsed, predicted_heads

    def Train(self, niters, training_data, test_data=None):
        """Train the arc perceptron."""
        for i in range(niters):
            print("\n**************-------------------*************")
            logging.info("Starting Training Epoch {}".format(i+1))
            #Train arc perceptron for one epoch.
            nr_correct_heads, nr_childs = self.arc_perceptron.Train(training_data)
            #Evaluate the arc perceptron
            if (i+1) % 5 == 0 or i+1 == niters:
                train_acc = self._Evaluate(training_data)
                logging.info("Train acc after iter {}: {}".format(i+1, train_acc))
            if test_data:
                logging.info("Evaluating on test data..")
                test_acc = self._Evaluate(test_data)
                # Comment out if you're not interested in seeing test acc after
                # each epoch.
                logging.info("Test acc after iter {}: {}".format(i+1, test_acc))
            #if train_acc == 100:
            #    break
            np.random.shuffle(training_data)
        self.train_acc = train_acc

    def _Evaluate(self, eval_data):
        """Evaluates the performance of arc perceptron on data.

        Args:
            eval_data = list, list of sentence_pb2.Sentence() objects.
        Returns:
            accuracy of the perceptron on the dataset.

        """
        acc = 0.0
        #print(len(eval_data))
        for sentence in eval_data:
            assert sentence.token[0].index == -1 and sentence.token[-1].index == -2
            assert sentence.HasField("length"), "Sentence must have a length!"
            #common.PPrintTextProto(sentence)
            _, predicted_heads = self.Parse(sentence)
            # get the gold heads for tokens except for the dummy ones.
            gold_heads = [token.selected_head.address for token in sentence.token[1:-1]]
            #logging.info("predicted {}, gold {}".format(predicted_heads, gold_heads))
            assert len(predicted_heads) == len(gold_heads), """Number of predicted and
                gold heads don't match!!!"""
            acc += self._Accuracy(predicted_heads, gold_heads)
        return acc / len(eval_data)


    def _Accuracy(self, predicted, gold):
        return 100 * sum(ph == gh for ph, gh in zip(predicted, gold)) / len(predicted)


    def _Softmax(self, matrix):
      """Numpy softmax function (normalizes rows)."""
      matrix -= np.max(matrix, axis=1, keepdims=True)
      #print(matrix)
      matrix = np.exp(matrix)
      #print(matrix)
      return matrix / np.sum(matrix, axis=1, keepdims=True)

    def Save(self, path, train_data_path=None, test_data_path=None, feature_file=None,
    	nr_epochs=None, test_accuracy=None):
        assert isinstance(train_data_path, str), "Invalid Data Path!"
        assert isinstance(test_data_path, str), "Invalid Data Path!"
        assert isinstance(nr_epochs, int), "Invalid number of epochs!"
        assert isinstance(test_accuracy, dict), "Invalid data type for test accuracy!"
        self.test_accuracy = test_accuracy
        self.arc_perceptron.SaveModel(
            name=path, train_data_path=train_data_path, test_data_path=test_data_path,
            feature_file=feature_file, nr_epochs=nr_epochs,
            test_accuracy=self.test_accuracy
        )

    def Load(self, path):
        assert isinstance(path, str), "Invalid Path!"
        self.arc_perceptron.LoadModel(path)



def main():
    parser = DependencyParser()

if __name__ == "__main__":
    main()
